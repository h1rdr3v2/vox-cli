#!/bin/sh
# Prepare a release: set the version, commit, tag vVERSION, and point the
# Homebrew formula at that tag. Pushing is left to you.
#
#   scripts/release.sh 0.3.0
#   git push origin main v0.3.0
set -e
cd "$(dirname "$0")/.."

version="$1"
if [ -z "$version" ]; then
  echo "usage: scripts/release.sh VERSION" >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "Commit or stash your changes first." >&2
  exit 1
fi
if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
  echo "Tag v$version already exists." >&2
  exit 1
fi

# -i.bak works with both BSD (macOS) and GNU sed.
sed -i.bak "s/^version = \".*\"/version = \"$version\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$version\"/" src/vox/__init__.py
rm -f pyproject.toml.bak src/vox/__init__.py.bak
if ! git diff --quiet; then
  git commit -q -am "Release $version"
fi
git tag "v$version"

revision=$(git rev-parse HEAD)
sed -i.bak -e "s/tag:      \".*\"/tag:      \"v$version\"/" -e "s/revision: \".*\"/revision: \"$revision\"/" Formula/vox.rb
rm -f Formula/vox.rb.bak
git commit -q -am "Homebrew formula: vox $version"

echo "Tagged v$version at $revision and updated Formula/vox.rb."
echo "Publish with: git push origin main v$version"
