"""Compiler control-flow lowering tests."""

from postpython.compiler.frontend import compile_source
from postpython.compiler.ir import Branch, CondBranch


def compile_errors(source: str) -> list:
    _, errors = compile_source(source)
    return errors


def branch_targets(source: str) -> list[str]:
    module, errors = compile_source(source)
    assert errors == []
    targets: list[str] = []
    for block in module.functions[0].blocks:
        term = block.terminator
        if isinstance(term, Branch):
            targets.append(term.target)
        elif isinstance(term, CondBranch):
            targets.extend([term.true_target, term.false_target])
    return targets


def test_break_in_for_loop_branches_to_after_block():
    source = """\
def f(n: int) -> int:
    total: int = 0
    for i in range(n):
        if i == 3:
            break
        total += i
    return total
"""

    assert any(target.startswith("for_after") for target in branch_targets(source))


def test_continue_in_for_loop_branches_to_step_block():
    source = """\
def f(n: int) -> int:
    total: int = 0
    for i in range(n):
        if i == 3:
            continue
        total += i
    return total
"""

    assert any(target.startswith("for_step") for target in branch_targets(source))


def test_break_and_continue_in_while_loop_are_lowered():
    source = """\
def f(n: int) -> int:
    i: int = 0
    while i < n:
        i += 1
        if i == 2:
            continue
        if i == 4:
            break
    return i
"""

    targets = branch_targets(source)
    assert any(target.startswith("while_cond") for target in targets)
    assert any(target.startswith("while_after") for target in targets)


def test_with_statement_reports_compiler_error_instead_of_being_dropped():
    source = """\
def f(x: int) -> int:
    with manager():
        x += 1
    return x
"""

    errors = compile_errors(source)
    assert [error.code for error in errors] == ["PP101"]
    assert "`with` statements" in errors[0].message


def test_try_statement_reports_compiler_error_instead_of_being_dropped():
    source = """\
def f(x: int) -> int:
    try:
        x += 1
    except Exception:
        x += 2
    return x
"""

    errors = compile_errors(source)
    assert [error.code for error in errors] == ["PP101"]
    assert "`try` statements" in errors[0].message


def test_match_statement_reports_compiler_error_instead_of_being_dropped():
    source = """\
def f(x: int) -> int:
    match x:
        case 0:
            return 1
        case _:
            return x
"""

    errors = compile_errors(source)
    assert [error.code for error in errors] == ["PP101"]
    assert "`match` statements" in errors[0].message


def test_comprehension_reports_compiler_error_instead_of_being_dropped():
    source = """\
def f(n: int) -> int:
    values: int = [i for i in range(n)]
    return n
"""

    errors = compile_errors(source)
    assert [error.code for error in errors] == ["PP101"]
    assert "comprehensions" in errors[0].message


def test_non_range_for_loop_reports_compiler_error_instead_of_being_dropped():
    source = """\
from postyp import Array, Int64

def f(xs: Array[Int64]) -> int:
    total: int = 0
    for x in xs:
        total += x
    return total
"""

    errors = compile_errors(source)
    assert [error.code for error in errors] == ["PP101"]
    assert "`for` loops over non-range iterables" in errors[0].message
