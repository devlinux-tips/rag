# SurrealDB Production Research

This research folder contains comprehensive analysis and practical examples for deploying SurrealDB in production environments with proper authentication and security.

## 📁 Contents

### Documentation
- **`production-deployment-analysis.md`** - Comprehensive analysis of production deployment options, storage backends, authentication mechanisms, and security best practices

### Deployment Configurations
- **`docker-compose.production.yml`** - Production-ready Docker Compose setup with security hardening, monitoring, and SSL proxy
- **`kubernetes-production.yaml`** - Full Kubernetes deployment with TiKV backend, security policies, auto-scaling, and monitoring
- **`setup-production.sh`** - Automated production setup script for systemd service, nginx proxy, and monitoring

### Testing
- **`simple-test.py`** - Python test suite to validate SurrealDB connectivity, authentication, and basic operations

## 🎯 Key Research Findings

### Storage Backend Options

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **RocksDB** | Single node production | High performance, stable | Not distributed |
| **TiKV** | Distributed/Kubernetes | Cloud-native, ACID transactions | Complex setup |
| **SurrealKV** | Native deployments | Optimized for SurrealDB | Newer, less tested |
| **Memory** | Development/testing | Extremely fast | No persistence |

### Production Authentication Architecture

```
ROOT USER (System Admin)
├── NAMESPACE USERS (Tenant Admins)
├── DATABASE USERS (App Admins)
└── RECORD ACCESS (End Users)
```

**Key Security Features:**
- Authentication enabled by default (v2.0+)
- Capability-based security (deny-all + allow-specific)
- Session duration management (5d users, 12h tokens)
- WebSocket connection isolation per user

### Deployment Recommendations

#### Small to Medium Applications
```bash
# Docker with RocksDB
docker run -p 8000:8000 -v /data:/data \
  surrealdb/surrealdb:latest start \
  --user admin --pass secret \
  --deny-all --allow-funcs "array,string,crypto::argon2" \
  rocksdb:/data/production.db
```

#### Large Scale/Distributed
- **Kubernetes + TiKV** for horizontal scaling
- **Load balancer** with SSL termination
- **Network policies** for security isolation
- **Auto-scaling** based on CPU/memory metrics

#### Managed Service
- **Surreal Cloud** for reduced operational overhead
- Built-in monitoring and backup
- Global edge deployment

## 🚀 Quick Start

### 1. Local Testing
```bash
# Install SurrealDB Python client
pip install surrealdb

# Run local SurrealDB (simple)
docker run -p 8000:8000 surrealdb/surrealdb:latest start --user admin --pass password memory

# Test connectivity
python simple-test.py
```

### 2. Production Docker Setup
```bash
# Create secrets
mkdir -p secrets
echo "your-secure-password" > secrets/surrealdb_password.txt

# Start production stack
docker-compose -f docker-compose.production.yml up -d

# Test production setup
python simple-test.py http://localhost:8000 admin your-secure-password
```

### 3. Kubernetes Deployment
```bash
# Apply Kubernetes configuration
kubectl apply -f kubernetes-production.yaml

# Check deployment status
kubectl get pods -n surrealdb-production

# Port forward for testing
kubectl port-forward -n surrealdb-production svc/surrealdb 8000:8000

# Test connection
python simple-test.py http://localhost:8000 admin your-secure-password
```

### 4. Automated Production Setup
```bash
# Run automated setup script
chmod +x setup-production.sh
./setup-production.sh

# Start service
sudo systemctl start surrealdb
sudo systemctl status surrealdb
```

## 🔒 Security Best Practices

### 1. Authentication Setup
```sql
-- Create production users with limited scope
DEFINE USER api_user ON DATABASE PASSWORD 'secure_password' DURATION FOR SESSION 12h;

-- Define record-level access
DEFINE ACCESS user_access ON DATABASE TYPE RECORD
SIGNUP (CREATE user SET email = $email, pass = crypto::argon2::generate($pass))
SIGNIN (SELECT * FROM user WHERE email = $email AND crypto::argon2::compare(pass, $pass))
DURATION FOR SESSION 7d, FOR TOKEN 1h;
```

### 2. Capability Restrictions
```bash
# Minimal capabilities for production
--deny-all
--allow-funcs "array,string,crypto::argon2,math,time"
--allow-net "api.internal.com:443"
```

### 3. Network Security
- TLS/SSL termination at load balancer
- Internal network communication only
- Firewall rules limiting access
- VPN/VPC for administrative access

## 📊 Monitoring & Backup

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# Database info query
curl -X POST http://localhost:8000/sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "INFO FOR DB;"}'
```

### Automated Backups
```bash
# Daily backup script (included in setup-production.sh)
/usr/local/bin/surrealdb-backup.sh

# Manual backup
surreal export --conn http://localhost:8000 \
  --user admin --pass password \
  --ns production --db main backup.sql
```

## 🏗️ Architecture Patterns

### High Availability Setup
1. **Load Balancer** - Multiple SurrealDB instances
2. **Shared Storage** - TiKV for distributed persistence
3. **Monitoring** - Prometheus + Grafana
4. **Backup Strategy** - Automated daily backups with retention

### Microservices Integration
- Service mesh compatibility
- API gateway integration
- JWT token validation
- Multi-tenant data isolation

## ❓ Common Issues & Solutions

### Authentication Problems
- **Issue**: "There was a problem with authentication"
- **Solution**: Ensure user exists and password is correct, check session duration

### Connection Issues
- **Issue**: Connection timeout or refused
- **Solution**: Check firewall rules, verify bind address (0.0.0.0:8000 for external access)

### Storage Backend Migration
- **Issue**: Changing storage backends
- **Solution**: Export data, recreate with new backend, import data

### Performance Optimization
- **Issue**: Slow queries or high memory usage
- **Solution**: Use appropriate storage backend, optimize queries, configure resource limits

## 📚 Additional Resources

- [SurrealDB Official Documentation](https://surrealdb.com/docs)
- [Security Best Practices](https://surrealdb.com/docs/surrealdb/reference-guide/security-best-practices)
- [Kubernetes Deployment Guide](https://surrealdb.com/docs/surrealdb/deployment/kubernetes)
- [Docker Running Guide](https://surrealdb.com/docs/surrealdb/installation/running/docker)

---

**Research Conducted**: September 2025
**Focus**: Production deployment with proper authentication
**Target**: Real-world production environments, not development setups