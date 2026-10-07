"""Bounded equation proofs, never evaluation of model-provided code.

Raw operands must occur in the cited source. Derived operands may use only
earlier verified equations in the same answer. Exact and rounded results are
distinct; this verifies arithmetic, not semantic choice of rows or methods.
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import re

_NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3}){1,4}|\d{1,15})(?:\.\d{1,12})?(?:[KMB%])?'
_SOURCE_NUMBER = re.compile(rf'(?<![A-Za-z0-9.,])({_NUMBER})(?![A-Za-z0-9]|[.,]\d)')
# Overlapping matches retain the second binary expression in A/B=C/D=result.
# This remains a bounded binary proof, not arbitrary expression evaluation.
_EQUATION = re.compile(rf'(?=(?<![A-Za-z0-9.,])(?:\(\s*)?({_NUMBER})\s*([-−+*/÷×])\s*({_NUMBER})\s*\)?\s*(=|≈)\s*({_NUMBER})(?![A-Za-z0-9]|[.,]\d))')
_SCALES = {'K': Decimal(1000), 'M': Decimal(1000000), 'B': Decimal(1000000000)}


def _value(token):
    token = token.replace(',', '')
    return Decimal(token[:-1]) * _SCALES[token[-1]] if token[-1] in _SCALES else Decimal(token.rstrip('%'))


def verified_equation_numbers(answer: str, evidence: str) -> set[str]:
    proven: set[str] = set()
    source_values = {(_value(match.group(1)), match.group(1).endswith('%'))
                     for match in _SOURCE_NUMBER.finditer(evidence)}
    def supported(token):
        return token in proven or bool(re.search(r'(?<![A-Za-z0-9.])' + re.escape(token)
            + r'(?![A-Za-z0-9.])', evidence)) or (_value(token),token.endswith('%')) in source_values
    answer = answer.replace('**','').replace('`','')
    with localcontext() as ctx:
        ctx.prec = 48
        for match in list(_EQUATION.finditer(answer))[:24]:
            left, operation, right, equality, result = match.groups()
            if not supported(left) or not supported(right):
                continue
            try:
                a, b, claimed = _value(left), _value(right), _value(result)
                if operation in ('/', '÷') and b == 0:
                    continue
                if operation in ('-', '−'):
                    actual = a - b
                elif operation == '+':
                    actual = a + b
                elif operation in ('*', '×'):
                    actual = a * b
                else:
                    actual = a / b
                # Percent/scalar operands cannot silently change units.
                if left.endswith('%') != right.endswith('%'):
                    continue
                if result.endswith('%') and not (left.endswith('%') and operation in ('-', '−', '+')):
                    continue
                if equality == '≈':
                    if result[-1] in _SCALES:
                        continue
                    quantum = Decimal(1).scaleb(Decimal(result.rstrip('%')).as_tuple().exponent)
                    actual = actual.quantize(quantum, rounding=ROUND_HALF_UP)
                if actual == claimed:
                    # Expanded operands are only admitted in a correct proof
                    # whose values are equivalent to the cited raw quantities.
                    proven.update((left,right,result))
            except (InvalidOperation, ArithmeticError):
                continue
    return proven
