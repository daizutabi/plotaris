# plotaris

[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Build Status][GHAction-image]][GHAction-link]
[![Coverage Status][codecov-image]][codecov-link]
[![Python Version][python-v-image]][python-v-link]

**plotaris** is a high-level plotting library for [Polars](https://github.com/pola-rs/polars), designed to bridge the gap between powerful data processing and elegant visualization. It provides advanced faceting capabilities for Matplotlib and helper channels for Altair, enabling a seamless workflow for data scientists.

## Key Features

- **Advanced Faceting for Matplotlib**: A powerful `FacetGrid` implementation optimized for Polars DataFrames, allowing for complex multi-plot layouts with ease.
- **Altair-inspired Channels**: Simplified API for Altair using `X`, `Y`, `Color`, and other channels, with built-in support for unit formatting and scientific notation.
- **Clean and Intuitive API**: Designed for readability and method chaining, making your plotting code more expressive.
- **Type-Safe**: Built with modern Python features and fully type-hinted for a better developer experience.

## Installation

```bash
pip install plotaris
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- Badges -->

[pypi-v-image]: https://img.shields.io/pypi/v/plotaris.svg
[pypi-v-link]: https://pypi.org/project/plotaris/
[GHAction-image]: https://github.com/daizutabi/plotaris/actions/workflows/ci.yaml/badge.svg?branch=main&event=push
[GHAction-link]: https://github.com/daizutabi/plotaris/actions?query=event%3Apush+branch%3Amain
[codecov-image]: https://codecov.io/github/daizutabi/plotaris/graph/badge.svg?token=Yu6lAdVVnd
[codecov-link]: https://codecov.io/github/daizutabi/plotaris?branch=main
[python-v-image]: https://img.shields.io/pypi/pyversions/plotaris.svg
[python-v-link]: https://pypi.org/project/plotaris
