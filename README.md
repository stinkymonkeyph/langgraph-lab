
# Local Setup Guide

This guide will help you set up the project locally.

## Prerequisites

- Ensure you have **Python 3.12.5** installed.
- Install **Pipenv** for managing dependencies.

## Install Pipenv

If you haven't installed **Pipenv**, run:

```sh
pip install --user pipenv
```

## Activate Virtual Environment

Enter the Pipenv shell:

```sh
pipenv shell
```

## Install Dependencies

To install all required dependencies, run:

```sh
pipenv install
```

## Setup Environment Variables

Before running the project, set up the required environment variables.

```sh
export OPENAI_API_KEY="your-api-key-here"
```

