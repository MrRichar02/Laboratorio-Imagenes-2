{
  description = "Entorno de desarrollo para el laboratorio numero 2 de PDI";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python312Packages.uv
        ruff
        basedpyright
      ];
      shellHook = ''
        unset PYTHONPATH
        uv sync
        source .venv/bin/activate
      '';
    };
  };
}
