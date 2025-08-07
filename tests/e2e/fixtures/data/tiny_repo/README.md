# Tiny Test Repository

A minimal repository for E2E testing GitHub analysis functionality.

## Features

- Basic Python application structure
- Security vulnerabilities for testing detection
- CI/CD workflow examples
- Documentation structure

## Known Vulnerabilities (Intentional for Testing)

This repository contains intentional security issues for testing:

1. Hardcoded credentials in config
2. Insecure random number generation  
3. SQL injection vulnerability
4. Unvalidated input handling

## Usage

```bash
pip install -r requirements.txt
python src/main.py
```

## Cost Estimation Context

- Expected cloud hosting: AWS ECS
- Estimated monthly cost: $150-300
- Traffic: ~1000 requests/day
- Database: PostgreSQL RDS (db.t3.micro)