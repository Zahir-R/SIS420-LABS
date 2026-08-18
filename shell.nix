let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/b5aa0fbd538984f6e3d201be0005b4463d8b09f8.tar.gz";
    sha256 = "sha256-oPXCU/SSUokcGaJREHibG1CBX3+s/W7orDWQOZDsEeQ=";
  }) { };
in
pkgs.mkShell {
  name = "ai1";
  buildInputs = with pkgs; [ python311 ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];
}
