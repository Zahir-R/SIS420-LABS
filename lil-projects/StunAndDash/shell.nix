let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/b5aa0fbd538984f6e3d201be0005b4463d8b09f8.tar.gz";
    sha256 = "sha256-oPXCU/SSUokcGaJREHibG1CBX3+s/W7orDWQOZDsEeQ=";
  }) { };
in
pkgs.mkShell {
  name = "stun-and-dash";
  buildInputs = with pkgs; [
    python311
    uv
    nodejs_22
  ];
  shellHook = ''
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
    export UV_PYTHON_PREFERENCE=only-system
    export PYTHONPATH="$PWD/python:$PYTHONPATH"
    export PATH="$PWD/web/node_modules/.bin:$PATH"
    echo "Stun & Dash: Python $(python --version 2>&1), Node $(node --version)"
    echo "Python: uv run python ... | Web: cd web && npm install && npm run dev"
  '';
}
