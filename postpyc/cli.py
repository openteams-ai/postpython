"""POST Python command-line interface.

    post-py check FILE...          structural subset checking
    post-py build FILE [options]   compile to native artifacts

The build subcommand is the stable entry point for packaging recipes
(conda/pixi, nix, spack). ``--prefix`` produces the ``libpp<name>``
install layout described in docs/distribution.md:

    $PREFIX/lib/lib<artifact>.so            (.dylib on macOS)
    $PREFIX/include/<artifact>.h
    $PREFIX/share/post-py/<artifact>.json
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Optional, Sequence

from .build import BuildError, build_file
from .checker import check_file

_LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"


def _cmd_check(args: argparse.Namespace) -> int:
    total = 0
    for path in args.files:
        violations = check_file(path)
        for violation in violations:
            print(violation)
        total += len(violations)
    if total:
        print(f"\n{total} violation(s) found.", file=sys.stderr)
        return 1
    return 0


def _artifact_name(path: Path, module_name: Optional[str]) -> str:
    if module_name:
        return module_name
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved.name
    if resolved.stem in ("__init__", "__post__"):
        return resolved.parent.name
    return resolved.stem


def _cmd_build(args: argparse.Namespace) -> int:
    path = Path(args.file)
    artifact = _artifact_name(path, args.module_name)
    kwargs = dict(
        cc=args.cc,
        cflags=args.cflags,
        keep_c=args.keep_c,
        cross_module_inline=args.cross_module_inline,
        search_paths=[Path(p) for p in args.search_path] or None,
    )

    try:
        if args.prefix is not None:
            if args.ext_module:
                print("error: --prefix and --ext-module are separate build "
                      "targets; build the extension with --output into the "
                      "recipe's site-packages instead", file=sys.stderr)
                return 2
            prefix = Path(args.prefix)
            lib_dir = prefix / "lib"
            include_dir = prefix / "include"
            share_dir = prefix / "share" / "post-py"
            for directory in (lib_dir, include_dir, share_dir):
                directory.mkdir(parents=True, exist_ok=True)

            lib_path = build_file(
                path,
                output=lib_dir / f"lib{artifact}{_LIB_SUFFIX}",
                emit_header=True,
                emit_manifest=True,
                module_name=artifact,
                **kwargs,
            )
            header = lib_path.with_suffix(".h")
            manifest = lib_path.with_suffix(".json")
            header.replace(include_dir / f"{artifact}.h")
            manifest.replace(share_dir / f"{artifact}.json")
            print(lib_path)
            print(include_dir / f"{artifact}.h")
            print(share_dir / f"{artifact}.json")
            return 0

        output = build_file(
            path,
            output=Path(args.output) if args.output else None,
            ext_module=args.ext_module,
            module_name=args.module_name,
            emit_header=args.emit_header,
            emit_manifest=args.emit_manifest,
            **kwargs,
        )
        print(output)
        if args.emit_header:
            print(output.with_suffix(".h"))
        if args.emit_manifest:
            print(output.with_suffix(".json"))
        return 0
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="POST Python reference compiler toolchain.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check files against the POST Python subset")
    check.add_argument("files", nargs="+", metavar="FILE")
    check.set_defaults(func=_cmd_check)

    build = sub.add_parser(
        "build",
        help="compile a POST Python file (and its POST imports) to native code",
    )
    build.add_argument("file", metavar="FILE")
    build.add_argument("--output", "-o", help="output artifact path")
    build.add_argument(
        "--prefix",
        help="install the libpp<name> layout (lib/, include/, share/post-py/) "
             "under this prefix",
    )
    build.add_argument("--ext-module", action="store_true",
                       help="build an importable CPython extension module")
    build.add_argument("--module-name", help="artifact / importable module name")
    build.add_argument("--emit-header", action="store_true",
                       help="write the C ABI header next to the output")
    build.add_argument("--emit-manifest", action="store_true",
                       help="write the JSON export manifest next to the output")
    build.add_argument("--search-path", action="append", default=[],
                       metavar="DIR", help="additional POST module source root")
    build.add_argument(
        "--cross-module-inline", action="store_true",
        help="replicate imported POST functions as static-inline copies in "
             "importing translation units so the C compiler can inline "
             "across module boundaries (public symbols are unchanged)",
    )
    build.add_argument("--cc", default="cc", help="C compiler (default: cc)")
    build.add_argument("--cflags", nargs="*", default=None,
                       help="extra flags for compile and link")
    build.add_argument("--keep-c", action="store_true",
                       help="keep intermediate C and object files")
    build.set_defaults(func=_cmd_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
