# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Functional tests for ``--exclude``/``--include`` filter behavior.

Covers issues #1117 (excluded unreadable files raising rc=2) and #1138
(traversal cost on excluded subtrees), and exercises filter semantics
across the four S3 transfer commands (sync, cp, mv, rm).
"""
import os

from awscli.testutils import mock, skip_if_windows
from tests.functional.s3 import BaseS3TransferCommandTest


class _BaseFilterTraverse(BaseS3TransferCommandTest):
    """Shared helpers for the four command-specific test classes."""

    def _put_object_response(self):
        return {'ETag': '"c8afdb36c52cf4727836669019e69222"'}

    def _list_dest(self):
        """Mock responses for any list-the-destination calls a command makes
        before transferring. Subclasses override based on command semantics.

        - ``s3 sync`` lists the destination bucket once.
        - ``s3 cp``/``s3 mv`` upload directly without listing the destination.
        """
        return []

    def _operation_keys(self, op_name):
        keys = []
        for op in self.operations_called:
            if op[0].name == op_name:
                keys.append(op[1]['Key'].replace('\\', '/'))
        return sorted(keys)

    def _uploaded_keys(self):
        return self._operation_keys('PutObject')

    def _deleted_keys(self):
        return self._operation_keys('DeleteObject')


class _LocalToS3Mixin:
    """For sync/cp/mv: source is the local rootdir, dest is s3://bucket/."""

    def _run(self, args, expected_rc=0):
        cmdline = '%s %s s3://bucket/ %s' % (
            self.prefix, self.files.rootdir, args,
        )
        return self.run_cmd(cmdline, expected_rc=expected_rc)


class _UploadFilterCases:
    """Test methods reused by sync/cp/mv (all local→s3 with traversal).

    Mixed into each command class so failures are reported per-command.
    """

    @skip_if_windows('POSIX-only file mode test.')
    def test_unreadable_file_in_excluded_dir_does_not_fail(self):
        """#1117: 0o000 file in an excluded directory must not affect rc.

        Also asserts the path is absent from stderr — a clean fix prunes
        the file before the warning machinery runs, so no message exists.
        """
        bad_path = self.files.create_file(
            os.path.join('excluded', 'bad'), 'data')
        os.chmod(bad_path, 0o000)
        try:
            self.parsed_responses = self._list_dest()
            _, stderr, _ = self._run(
                "--only-show-errors --exclude excluded/*", expected_rc=0)
            self.assertNotIn('excluded/bad', stderr)
            self.assertNotIn('not readable', stderr)
        finally:
            os.chmod(bad_path, 0o600)

    @skip_if_windows('POSIX-only directory mode test.')
    def test_unreadable_directory_inside_excluded_path_not_descended(self):
        """#1138: listdir must not be called on excluded subtrees.

        Also asserts the path is absent from stderr.
        """
        self.files.create_file(
            os.path.join('excluded', 'sub', 'inner.txt'), 'data')
        sub_path = os.path.join(self.files.rootdir, 'excluded', 'sub')
        os.chmod(sub_path, 0o000)
        try:
            self.parsed_responses = self._list_dest()
            _, stderr, _ = self._run(
                "--only-show-errors --exclude excluded/*", expected_rc=0)
            self.assertNotIn('excluded/sub', stderr)
            self.assertNotIn('not readable', stderr)
        finally:
            os.chmod(sub_path, 0o755)

    def test_filter_excludes_everything_no_uploads(self):
        """``--exclude *`` alone yields rc=0 with no uploads."""
        self.files.create_file('a.txt', 'data')
        self.files.create_file(os.path.join('sub', 'b.txt'), 'data')

        self.parsed_responses = self._list_dest()
        self._run("--exclude *", expected_rc=0)
        self.assertEqual(self._uploaded_keys(), [])

    def test_empty_source_directory_no_uploads(self):
        """Sync of an empty directory yields rc=0 with no uploads."""
        self.parsed_responses = self._list_dest()
        self._run("", expected_rc=0)
        self.assertEqual(self._uploaded_keys(), [])

    def test_multiple_includes_combine_with_exclude_star(self):
        """``--exclude * --include *.txt --include *.md`` keeps both kinds."""
        self.files.create_file('a.txt', 'data')
        self.files.create_file('b.md', 'data')
        self.files.create_file('c.log', 'data')
        self.files.create_file('d.bin', 'data')

        self.parsed_responses = self._list_dest() + [
            self._put_object_response(),
            self._put_object_response(),
        ]
        self._run("--exclude * --include *.txt --include *.md",
                  expected_rc=0)
        self.assertEqual(self._uploaded_keys(), ['a.txt', 'b.md'])

    def test_deep_include_with_exclude_star_uploads_only_target(self):
        """``--exclude * --include a/b/c.txt`` uploads exactly c.txt."""
        self.files.create_file(os.path.join('a', 'b', 'c.txt'), 'data')
        self.files.create_file(os.path.join('a', 'b', 'x.txt'), 'data')
        self.files.create_file(os.path.join('a', 'y.txt'), 'data')
        self.files.create_file('z.txt', 'data')

        self.parsed_responses = self._list_dest() + [
            self._put_object_response()]
        self._run("--exclude * --include a/b/c.txt", expected_rc=0)
        self.assertEqual(self._uploaded_keys(), ['a/b/c.txt'])

    def test_include_overrides_exclude_in_same_subdirectory(self):
        """``--exclude cache/* --include cache/keep.txt`` keeps the target."""
        self.files.create_file(os.path.join('cache', 'keep.txt'), 'data')
        self.files.create_file(os.path.join('cache', 'drop.txt'), 'data')

        self.parsed_responses = self._list_dest() + [
            self._put_object_response()]
        self._run(
            "--exclude cache/* --include cache/keep.txt", expected_rc=0)
        self.assertEqual(self._uploaded_keys(), ['cache/keep.txt'])

    def test_include_with_mid_path_glob(self):
        """``--include a/*/c.txt`` matches any single-segment middle dir."""
        self.files.create_file(os.path.join('a', 'x', 'c.txt'), 'data')
        self.files.create_file(os.path.join('a', 'y', 'c.txt'), 'data')
        self.files.create_file(os.path.join('a', 'x', 'd.txt'), 'data')

        self.parsed_responses = self._list_dest() + [
            self._put_object_response(),
            self._put_object_response(),
        ]
        self._run("--exclude * --include a/*/c.txt", expected_rc=0)
        self.assertEqual(self._uploaded_keys(), ['a/x/c.txt', 'a/y/c.txt'])

    def test_pattern_order_last_match_wins(self):
        """Filters listed later override earlier ones."""
        included = [
            os.path.join('included', 'foo'),
            os.path.join('included', 'bar'),
            os.path.join('excluded', 'included', 'foo'),
            os.path.join('excluded', 'included', 'bar'),
        ]
        excluded = [
            os.path.join('excluded', 'foo'),
            os.path.join('excluded', 'bar'),
            os.path.join('included', 'excluded', 'foo'),
            os.path.join('included', 'excluded', 'bar'),
        ]
        for path in included + excluded:
            self.files.create_file(path, 'data')

        self.parsed_responses = self._list_dest() + [
            self._put_object_response() for _ in included]

        self._run(
            "--exclude excluded/* "
            "--include excluded/included/* "
            "--exclude included/excluded/*",
            expected_rc=0,
        )
        expected_keys = sorted(p.replace(os.sep, '/') for p in included)
        self.assertEqual(self._uploaded_keys(), expected_keys)

    def test_glob_include_under_exclude_star_finds_all_matches(self):
        """Regression case from PR #5425 ([discussion_r883991435]).

        ``--exclude '*' --include '*.py'`` must traverse subdirectories
        to locate ``.py`` files at every depth. A naive prune that
        skips every subtree matched by ``--exclude '*'`` would silently
        upload only the root-level ``.py`` and miss the nested ones.
        """
        self.files.create_file('top.py', 'data')
        self.files.create_file(os.path.join('directory', 'mid.py'), 'data')
        self.files.create_file(
            os.path.join('directory', 'another-dir', 'deep.py'), 'data')
        self.files.create_file(os.path.join('directory', 'note.txt'), 'data')

        self.parsed_responses = self._list_dest() + [
            self._put_object_response() for _ in range(3)]
        self._run("--exclude * --include *.py", expected_rc=0)
        self.assertEqual(
            self._uploaded_keys(),
            ['directory/another-dir/deep.py',
             'directory/mid.py',
             'top.py'])

    @skip_if_windows('Symlink semantics differ on Windows.')
    def test_broken_symlink_warning_survives_specific_exclude(self):
        """A broken symlink encountered during walking must still produce
        a 'does not exist' warning. The filter is for selecting which
        transfers to perform, not for swallowing fs-level validation
        errors.

        Uses a name-specific exclude (rather than ``--exclude '*'``)
        because ``*`` would prune the rootdir itself and the walker
        would never reach the symlink. With a name-specific pattern,
        the rootdir survives, the walker calls ``should_ignore_file``
        on the broken link, and the existence-before-filter ordering
        is what guarantees the warning surfaces.
        """
        # A real keep file so the rootdir is non-empty and not pruned.
        self.files.create_file('keep.txt', 'data')
        target = os.path.join(self.files.rootdir, 'no_such_target')
        link = os.path.join(self.files.rootdir, 'broken_link')
        os.symlink(target, link)

        self.parsed_responses = self._list_dest() + [
            self._put_object_response()]
        _, stderr, _ = self._run("--exclude broken_link", expected_rc=2)
        self.assertIn('does not exist', stderr)

    @skip_if_windows('Uses POSIX path realpath comparison.')
    def test_excluded_subtree_is_not_listdired(self):
        """Direct prune evidence for #1138.

        Spies on ``os.listdir`` inside the file generator and asserts
        the excluded subtree is never listed. Side-effect-only tests
        (rc / uploaded keys) cannot distinguish between "filter dropped
        late" and "walker pruned early"; this one can.
        """
        self.files.create_file(
            os.path.join('excluded', 'a.txt'), 'data')
        self.files.create_file(os.path.join('kept', 'b.txt'), 'data')
        excluded_path = os.path.realpath(
            os.path.join(self.files.rootdir, 'excluded'))

        real_listdir = os.listdir
        listed_paths = []

        def spy(path):
            listed_paths.append(os.path.realpath(path))
            return real_listdir(path)

        self.parsed_responses = self._list_dest() + [
            self._put_object_response()]
        with mock.patch(
                'awscli.customizations.s3.filegenerator.os.listdir',
                side_effect=spy):
            self._run("--exclude excluded/*", expected_rc=0)

        self.assertNotIn(excluded_path, listed_paths)
        self.assertEqual(self._uploaded_keys(), ['kept/b.txt'])


class TestSyncFilterTraverse(
        _UploadFilterCases, _LocalToS3Mixin, _BaseFilterTraverse):
    prefix = 's3 sync '

    def _list_dest(self):
        # sync compares destination by listing it once.
        return [self.list_objects_response([])]


class TestCpFilterTraverse(
        _UploadFilterCases, _LocalToS3Mixin, _BaseFilterTraverse):
    prefix = 's3 cp --recursive '


class TestMvFilterTraverse(
        _UploadFilterCases, _LocalToS3Mixin, _BaseFilterTraverse):
    prefix = 's3 mv --recursive '


class TestSyncDownloadFilterTraverse(_BaseFilterTraverse):
    """``s3 sync s3://b/ local/`` with filters applied to S3 keys."""
    prefix = 's3 sync '

    def _run(self, args, expected_rc=0):
        cmdline = '%s s3://bucket/ %s %s' % (
            self.prefix, self.files.rootdir, args,
        )
        return self.run_cmd(cmdline, expected_rc=expected_rc)

    def _downloaded_keys(self):
        return self._operation_keys('GetObject')

    def test_excluded_keys_not_downloaded(self):
        """``--exclude logs/*`` skips matching keys; rest are fetched."""
        self.parsed_responses = [
            self.list_objects_response(
                ['keep/a.txt', 'logs/x.log', 'logs/y.log', 'keep/b.txt']),
            self.get_object_response(),
            self.get_object_response(),
        ]
        self._run("--exclude logs/*", expected_rc=0)
        self.assertEqual(
            self._downloaded_keys(), ['keep/a.txt', 'keep/b.txt'])

    def test_deep_include_with_exclude_star_downloads_only_target(self):
        """``--exclude * --include a/b/c.txt`` fetches exactly that key."""
        self.parsed_responses = [
            self.list_objects_response(
                ['a/b/c.txt', 'a/b/x.txt', 'a/y.txt', 'z.txt']),
            self.get_object_response(),
        ]
        self._run("--exclude * --include a/b/c.txt", expected_rc=0)
        self.assertEqual(self._downloaded_keys(), ['a/b/c.txt'])

    def test_filter_excludes_everything_no_downloads(self):
        """``--exclude *`` alone yields rc=0 with no downloads."""
        self.parsed_responses = [
            self.list_objects_response(['a.txt', 'sub/b.txt']),
        ]
        self._run("--exclude *", expected_rc=0)
        self.assertEqual(self._downloaded_keys(), [])

    @skip_if_windows('POSIX-only directory mode test.')
    def test_excluded_dest_subtree_with_unreadable_dir_does_not_warn(self):
        """sync s3://b/ ./dst --exclude excluded/*: a 0o000 directory
        sitting inside the excluded destination subtree must not be
        listdired (and so must not produce a 'not readable' warning).

        Without the dst-side prune, the reverse walker descends into
        /dst/excluded because the source-rooted patterns don't match
        the destination path; once inside, is_readable() on the 0o000
        sub-dir fires triggers_warning() and the user gets rc=2 even
        though they explicitly excluded the whole subtree.
        """
        self.files.create_file(
            os.path.join('excluded', 'sub', 'inner.txt'), 'data')
        sub_path = os.path.join(self.files.rootdir, 'excluded', 'sub')
        os.chmod(sub_path, 0o000)
        try:
            self.parsed_responses = [self.list_objects_response([])]
            _, stderr, _ = self._run(
                "--only-show-errors --exclude excluded/*", expected_rc=0)
            self.assertNotIn('excluded/sub', stderr)
            self.assertNotIn('not readable', stderr)
        finally:
            os.chmod(sub_path, 0o755)


class _CrossBucketCases:
    """Test methods reused by cp/mv/sync s3→s3.

    Subclasses must define ``prefix`` and ``_run``/``_setup_responses``.
    """

    def _copied_keys(self):
        return self._operation_keys('CopyObject')

    def test_excluded_keys_not_copied(self):
        """``--exclude logs/*`` skips matching keys; others are copied."""
        all_keys = ['keep/a.txt', 'logs/x.log', 'logs/y.log', 'keep/b.txt']
        kept = ['keep/a.txt', 'keep/b.txt']
        self._setup_responses(all_keys, len(kept))
        self._run("--exclude logs/*", expected_rc=0)
        self.assertEqual(self._copied_keys(), sorted(kept))

    def test_deep_include_with_exclude_star_copies_only_target(self):
        """``--exclude * --include a/b/c.txt`` copies exactly that key."""
        all_keys = ['a/b/c.txt', 'a/b/x.txt', 'a/y.txt', 'z.txt']
        self._setup_responses(all_keys, 1)
        self._run("--exclude * --include a/b/c.txt", expected_rc=0)
        self.assertEqual(self._copied_keys(), ['a/b/c.txt'])

    def test_filter_excludes_everything_no_copies(self):
        """``--exclude *`` alone copies nothing."""
        self._setup_responses(['a.txt', 'sub/b.txt'], 0)
        self._run("--exclude *", expected_rc=0)
        self.assertEqual(self._copied_keys(), [])


class TestCpS3ToS3FilterTraverse(_CrossBucketCases, _BaseFilterTraverse):
    prefix = 's3 cp --recursive '

    def _run(self, args, expected_rc=0):
        cmdline = '%s s3://srcbucket/ s3://dstbucket/ %s' % (
            self.prefix, args,
        )
        return self.run_cmd(cmdline, expected_rc=expected_rc)

    def _setup_responses(self, all_keys, copy_count):
        self.parsed_responses = (
            [self.list_objects_response(all_keys)]
            + [self.copy_object_response() for _ in range(copy_count)]
        )


class TestMvS3ToS3FilterTraverse(_CrossBucketCases, _BaseFilterTraverse):
    prefix = 's3 mv --recursive '

    def _run(self, args, expected_rc=0):
        cmdline = '%s s3://srcbucket/ s3://dstbucket/ %s' % (
            self.prefix, args,
        )
        return self.run_cmd(cmdline, expected_rc=expected_rc)

    def _setup_responses(self, all_keys, copy_count):
        # mv = copy + delete src; one DeleteObject per copied key.
        self.parsed_responses = (
            [self.list_objects_response(all_keys)]
            + [self.copy_object_response() for _ in range(copy_count)]
            + [self.empty_response() for _ in range(copy_count)]
        )


class TestSyncS3ToS3FilterTraverse(_CrossBucketCases, _BaseFilterTraverse):
    prefix = 's3 sync '

    def _run(self, args, expected_rc=0):
        cmdline = '%s s3://srcbucket/ s3://dstbucket/ %s' % (
            self.prefix, args,
        )
        return self.run_cmd(cmdline, expected_rc=expected_rc)

    def _setup_responses(self, all_keys, copy_count):
        # sync lists both buckets; destination is empty so all kept keys copy.
        self.parsed_responses = (
            [self.list_objects_response(all_keys)]
            + [self.list_objects_response([])]
            + [self.copy_object_response() for _ in range(copy_count)]
        )


class TestRmFilterTraverse(_BaseFilterTraverse):
    """``s3 rm`` works on S3 objects only — no local traversal."""
    prefix = 's3 rm --recursive '

    def _run(self, args, expected_rc=0):
        cmdline = '%s s3://bucket/ %s' % (self.prefix, args)
        return self.run_cmd(cmdline, expected_rc=expected_rc)

    def test_excluded_keys_not_deleted(self):
        """``--exclude logs/*`` must keep matching keys, delete the rest."""
        self.parsed_responses = [
            self.list_objects_response(
                ['keep/a.txt', 'logs/x.log', 'logs/y.log', 'keep/b.txt']),
            self.empty_response(),
            self.empty_response(),
        ]
        self._run("--exclude logs/*", expected_rc=0)
        self.assertEqual(
            self._deleted_keys(), ['keep/a.txt', 'keep/b.txt'])

    def test_deep_include_with_exclude_star_deletes_only_target(self):
        """``--exclude * --include a/b/c.txt`` deletes exactly that key."""
        self.parsed_responses = [
            self.list_objects_response(
                ['a/b/c.txt', 'a/b/x.txt', 'a/y.txt', 'z.txt']),
            self.empty_response(),
        ]
        self._run("--exclude * --include a/b/c.txt", expected_rc=0)
        self.assertEqual(self._deleted_keys(), ['a/b/c.txt'])

    def test_pattern_order_last_match_wins(self):
        """Same alternating-stack semantics as for upload commands."""
        all_keys = [
            'included/foo',
            'included/bar',
            'excluded/foo',
            'excluded/bar',
            'excluded/included/foo',
            'excluded/included/bar',
            'included/excluded/foo',
            'included/excluded/bar',
        ]
        kept = [
            'included/foo',
            'included/bar',
            'excluded/included/foo',
            'excluded/included/bar',
        ]
        self.parsed_responses = (
            [self.list_objects_response(all_keys)]
            + [self.empty_response() for _ in kept]
        )
        self._run(
            "--exclude excluded/* "
            "--include excluded/included/* "
            "--exclude included/excluded/*",
            expected_rc=0,
        )
        self.assertEqual(self._deleted_keys(), sorted(kept))
