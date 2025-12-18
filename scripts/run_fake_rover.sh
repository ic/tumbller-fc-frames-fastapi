# Run the Mini App server

set -e

root_dir=$(realpath $0 | rev | cut -d / -f 3- | rev)

if [[ ! -d "${root_dir}/venv" ]]
then
  >&2 echo "Setup requirements no found. Perhaps need to run scripts/setup.sh?"
  exit 1
fi

pushd $root_dir
source venv/bin/activate
PYTHONPATH=. python dev/fake_rover.py
popd
