"""Run a simple demonstration of the Draw class."""

from rand_ai.draw import Draw


def main() -> None:
    """Create and display a default example draw."""
    draw = Draw(1, 2, 3, 4, 5, 6)
    print(draw.num1, draw.num2, draw.num3, draw.num4, draw.num5, draw.num6)


if __name__ == "__main__":
    main()
