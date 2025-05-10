def f_f(f: float) -> str:
    """
    Format a float for additive terms, adding a '+' sign if positive.

    Args:
        f (float): The number to format.

    Returns:
        str: Formatted string with '+' if f > 0, otherwise the number as is.
    """
    if f > 0:
        return f"+{f:g}"
    return f"{f:g}"


def f_p(f: float) -> str:
    """
    Format a float for use in exponents or powers, adding parentheses if negative.

    Args:
        f (float): The number to format.

    Returns:
        str: Formatted string with parentheses if f < 0, otherwise the number as is.
    """
    if f > 0:
        return f"{f:g}"
    return f"({f:g})"