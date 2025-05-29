{
  description = "Python 3.10 environment for ttdemo with rdkit and related packages";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs, ... }: let
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
    pythonEnv = pkgs.python3.withPackages (ps: with ps; [
      rdkit
      pandas
      rapidfuzz
      aiohttp
      tenacity
      pyarrow
      fastparquet
      seaborn
      matplotlib
      scikit-learn
      cython
      blosc2
    ]);
  in {
    devShells."x86_64-linux".default = pkgs.mkShell {
      buildInputs = [ pythonEnv ];
      shellHook = ''
        echo "Installing pip-only packages..."
        python3 -m pip install --quiet --no-cache-dir pubchempy FuzzyTM fastcluster
      '';
    };

    # This exposes the Python environment as a package for use in other flakes:
    packages."x86_64-linux".default = pythonEnv;
  };
}