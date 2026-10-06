# Homebrew formula for vox. This repo doubles as its tap:
#   brew tap h1rdr3v2/vox-cli https://github.com/h1rdr3v2/vox-cli
#   brew install h1rdr3v2/vox-cli/vox
class Vox < Formula
  desc "Local transcription (Whisper) and speech (Kokoro) with zero idle memory"
  homepage "https://github.com/h1rdr3v2/vox-cli"
  url "https://github.com/h1rdr3v2/vox-cli.git",
      tag:      "v0.2.2",
      revision: "2d90f24cb05ca9fdbf8dae5417013f0e34d17541"
  license "MIT"
  head "https://github.com/h1rdr3v2/vox-cli.git", branch: "main"

  depends_on "ffmpeg"
  depends_on "python@3.12"

  on_macos do
    depends_on arch: :arm64 # MLX needs Apple Silicon
  end

  def install
    python = formula_opt_bin("python@3.12")/"python3.12"
    system python, "-m", "venv", libexec
    pip = [libexec/"bin/python", "-m", "pip", "--disable-pip-version-check"]
    # vox itself is pure Python: build its wheel and install it without dependencies.
    system(*pip, "wheel", "--no-deps", "--no-cache-dir", "--wheel-dir", libexec/"wheels", buildpath)
    wheel = Dir[libexec/"wheels/*.whl"].first
    system(*pip, "install", "--no-deps", "--no-cache-dir", wheel)
    (libexec/"requirements.txt").write "vox-cli @ file://#{wheel}\n"
    bin.install_symlink libexec/"bin/vox"
  end

  # The dependencies (MLX and torch on macOS, CTranslate2 and ONNX Runtime on
  # Linux, spaCy) are prebuilt wheels from PyPI. Installed during `install`,
  # Homebrew would then rewrite their libraries, which fails for some and
  # breaks code signatures. Post-install runs after that and still writes
  # inside the keg, so `brew uninstall vox` removes everything.
  if respond_to?(:post_install_steps) # Homebrew 7 and later
    post_install_steps do
      run "{{libexec}}/bin/python",
          args:           ["-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                           "-r", "{{libexec}}/requirements.txt"],
          network_access: true,
          writable_paths: ["{{libexec}}"]
    end
  else
    def post_install
      system libexec/"bin/python", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
             "-r", libexec/"requirements.txt"
    end
  end

  def caveats
    <<~EOS
      vox installs no models. Pick them when you first use it, or pull ahead:
        vox models pull whisper-large-v3-turbo
        vox models pull kokoro-82m

      To remove vox together with its models, settings and right-click
      actions, run `vox uninstall` (it runs `brew uninstall vox` itself).
      `brew uninstall vox` alone leaves ~/.config/vox and ~/.cache/vox behind.
    EOS
  end

  test do
    assert_match(/^vox \d+\.\d+/, shell_output("#{bin}/vox --version"))
    ENV["VOX_CONFIG_DIR"] = testpath/"config"
    ENV["VOX_CACHE_DIR"] = testpath/"cache"
    assert_match "No models installed", shell_output("#{bin}/vox models list --installed 2>&1")
    assert_match "whisper-large-v3-turbo", shell_output("#{bin}/vox models list --available")
  end
end
