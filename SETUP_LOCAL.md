# RAG Platform - Local Server Installation Guide

This guide walks you through setting up the RAG platform natively on Ubuntu 24.04 LTS without Docker.

## System Requirements

- **OS**: Ubuntu 24.04 LTS (or compatible Linux distribution)
- **Architecture**: x86_64
- **RAM**: Minimum 8GB, recommended 16GB+ (your server: 243GB ✓)
- **CPU**: Minimum 4 cores, recommended 8+ (your server: 144 cores ✓)
- **Disk**: Minimum 50GB free space for data and models
- **Network**: Internet connection for downloading dependencies

## Pre-Installation Checklist

✅ **Already Installed** (verified on your server):
- Python 3.x
- Node.js 18+ and npm
- Git

🔧 **Will Be Installed**:
- PostgreSQL 16
- Redis 7
- Nginx
- Weaviate v1.33.0
- Python packages (from requirements.txt)
- Node.js packages (for web-api and web-ui)

## Installation Steps

### 1. Prepare the Environment

Ensure you're in the repository root with the virtual environment active:

```bash
cd /home/rag/src/rag
source venv/bin/activate  # If not already activated
```

### 2. Run the Installation Script

The `setup-local-server.sh` script automates the entire installation:

```bash
sudo ./setup-local-server.sh
```

**What the script does:**
1. Updates system packages
2. Installs PostgreSQL 16 (port 5434)
3. Installs Redis 7 (port 6379)
4. Installs Nginx (port 80)
5. Downloads and installs Weaviate v1.33.0 (ports 8080, 50051)
6. Installs Python dependencies from requirements.txt
7. Installs Node.js dependencies for web-api and web-ui
8. Creates systemd service files for all components
9. Configures Nginx as reverse proxy
10. Creates `.env.local` configuration file
11. Starts and enables all services

**Installation time**: ~10-15 minutes (depending on network speed)

### 3. Verify Installation

After installation completes, check service status:

```bash
sudo ./manage-services.sh status
```

Expected output:
```
=== Service Status ===

  ● postgresql: running
  ● redis-server: running
  ● weaviate: running
  ● rag-api: running
  ● web-api: running
  ● web-ui: running
  ● nginx: running
```

Perform health checks:

```bash
sudo ./manage-services.sh health
```

## Service Architecture

### Services and Ports

| Service | Port | Description |
|---------|------|-------------|
| **Nginx** | 80 | Reverse proxy (main entry point) |
| **Web UI** | 5173 | React frontend (via Nginx at /) |
| **Web API** | 3000 | Express.js API (via Nginx at /api/v1/) |
| **RAG API** | 8082 | FastAPI Python service (via Nginx at /rag/) |
| **Weaviate** | 8080 | Vector database (HTTP) |
| **Weaviate gRPC** | 50051 | Weaviate gRPC endpoint |
| **PostgreSQL** | 5434 | Relational database |
| **Redis** | 6379 | Cache and real-time features |

### Service Dependencies

```
nginx
 ├── web-ui (React)
 ├── web-api (Express) → rag-api, postgresql, redis
 └── rag-api (FastAPI) → weaviate, postgresql
```

### Data Directories

- **Weaviate data**: `/home/rag/src/rag/weaviate_data/`
- **PostgreSQL data**: `/var/lib/postgresql/16/main/`
- **Redis data**: `/var/lib/redis/`
- **Application data**: `/home/rag/src/rag/services/rag-service/data/`
- **Models**: `/home/rag/src/rag/services/rag-service/models/`
- **Logs**: `/home/rag/src/rag/logs/` and `journalctl`

## Service Management

Use the `manage-services.sh` helper script for common operations:

### Start/Stop/Restart

```bash
# Start all services
sudo ./manage-services.sh start

# Stop all services
sudo ./manage-services.sh stop

# Restart all services
sudo ./manage-services.sh restart

# Restart a specific service
sudo ./manage-services.sh restart-one weaviate
```

### Status and Health

```bash
# Check service status
sudo ./manage-services.sh status

# Perform health checks
sudo ./manage-services.sh health
```

### Logs

```bash
# View logs for a specific service
sudo ./manage-services.sh logs rag-api

# View logs with more lines
sudo ./manage-services.sh logs weaviate 100

# Follow all service logs in real-time
sudo ./manage-services.sh logs-all
```

### Auto-Start Configuration

```bash
# Enable auto-start on boot (default: enabled)
sudo ./manage-services.sh enable

# Disable auto-start on boot
sudo ./manage-services.sh disable
```

### Direct systemctl Commands

You can also use systemctl directly:

```bash
# Check individual service status
sudo systemctl status weaviate

# View service logs
sudo journalctl -u rag-api -f

# Restart a service
sudo systemctl restart web-api
```

## Configuration

### Environment Variables

Main configuration file: `/home/rag/src/rag/.env.local`

```bash
# View configuration
cat /home/rag/src/rag/.env.local

# Edit configuration
sudo nano /home/rag/src/rag/.env.local

# After changes, restart affected services
sudo ./manage-services.sh restart
```

### Database Configuration

**PostgreSQL**:
- Database: `ragdb`
- User: `raguser`
- Password: `x|B&h@p4F@o|k6t;~X]1A((Z.,RG`
- Port: `5434` (non-standard to avoid conflicts)

Connect to database:
```bash
psql -h localhost -p 5434 -U raguser -d ragdb
```

### Weaviate Configuration

**Performance settings** (optimized for your 243GB/144-core server):
- Memory limit: 200GB
- CPU cores: 140 (leaves 4 for OS)
- LSM max segment: 10GB
- Access strategy: mmap

Located in: `/etc/systemd/system/weaviate.service`

To modify:
```bash
sudo nano /etc/systemd/system/weaviate.service
sudo systemctl daemon-reload
sudo systemctl restart weaviate
```

### Nginx Configuration

Main config: `/etc/nginx/sites-available/rag-platform`

To modify:
```bash
sudo nano /etc/nginx/sites-available/rag-platform
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

## Accessing the Platform

### Web Interface

**Primary access** (via Nginx):
```
http://localhost
http://your-server-ip
```

**Direct service access**:
- Web UI: `http://localhost:5173`
- Web API: `http://localhost:3000`
- RAG API: `http://localhost:8082`
- Weaviate: `http://localhost:8080`

### CLI Access

Activate virtual environment and use the CLI:

```bash
cd /home/rag/src/rag
source venv/bin/activate

# Check system status
python rag.py --language hr status

# Process documents
python rag.py --tenant development --user admin --language hr process-docs data/development/users/admin/documents/hr/

# Query the system
python rag.py --tenant development --user admin --language hr query "Što je RAG?"

# List collections
python rag.py --language hr list-collections
```

## Security Considerations

### 🔒 **IMPORTANT**: Production Security

The installation uses development defaults. **Change these in production**:

1. **JWT Secret**: Edit `.env.local` and change `JWT_SECRET`
2. **Database Password**: Update PostgreSQL password
3. **Firewall**: Configure UFW to restrict access
4. **SSL/TLS**: Add HTTPS with Let's Encrypt
5. **API Keys**: Never commit API keys to version control

### Firewall Configuration (Optional)

```bash
# Enable firewall
sudo ufw enable

# Allow SSH (if using remote access)
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Check status
sudo ufw status
```

## Troubleshooting

### Services Won't Start

1. **Check service status**:
   ```bash
   sudo ./manage-services.sh status
   ```

2. **View error logs**:
   ```bash
   sudo journalctl -u <service-name> -n 50
   ```

3. **Common issues**:
   - **Port already in use**: Check with `netstat -tlnp | grep <port>`
   - **Permission denied**: Ensure services run as `rag` user
   - **Database connection failed**: Verify PostgreSQL is running

### Weaviate Memory Issues

If Weaviate fails to start or crashes:

1. **Check available memory**:
   ```bash
   free -h
   ```

2. **Reduce memory limit** in `/etc/systemd/system/weaviate.service`:
   ```bash
   Environment="GOMEMLIMIT=100GiB"  # Reduce from 200GiB
   ```

3. **Reload and restart**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart weaviate
   ```

### Python/Node Dependency Issues

**Reinstall Python dependencies**:
```bash
cd /home/rag/src/rag
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

**Reinstall Node dependencies**:
```bash
# Web API
cd /home/rag/src/rag/services/web-api
rm -rf node_modules package-lock.json
npm install

# Web UI
cd /home/rag/src/rag/services/web-ui
rm -rf node_modules package-lock.json
npm install
```

### Database Connection Issues

**Test PostgreSQL connection**:
```bash
pg_isready -h localhost -p 5434 -U raguser -d ragdb
```

**Reset database** (⚠️ destroys all data):
```bash
sudo -u postgres psql -c "DROP DATABASE ragdb;"
sudo -u postgres psql -c "DROP USER raguser;"
sudo -u postgres psql -c "CREATE USER raguser WITH PASSWORD 'x|B&h@p4F@o|k6t;~X]1A((Z.,RG';"
sudo -u postgres psql -c "CREATE DATABASE ragdb OWNER raguser;"
```

### Nginx 502 Bad Gateway

1. **Check upstream services**:
   ```bash
   sudo ./manage-services.sh health
   ```

2. **Verify service ports**:
   ```bash
   netstat -tlnp | grep -E "(3000|8082|5173)"
   ```

3. **Check Nginx error log**:
   ```bash
   sudo tail -f /var/log/nginx/rag-error.log
   ```

## Performance Tuning

### Weaviate Optimization

Your server has excellent resources. Fine-tune Weaviate:

```bash
# Edit service file
sudo nano /etc/systemd/system/weaviate.service

# Recommended settings for 243GB RAM / 144 cores:
Environment="GOMEMLIMIT=200GiB"
Environment="GOMAXPROCS=140"
Environment="PERSISTENCE_LSM_MAX_SEGMENT_SIZE=10GB"

# Restart after changes
sudo systemctl daemon-reload
sudo systemctl restart weaviate
```

### PostgreSQL Tuning

For large datasets, tune PostgreSQL:

```bash
sudo nano /etc/postgresql/16/main/postgresql.conf

# Recommended changes:
shared_buffers = 32GB
effective_cache_size = 180GB
maintenance_work_mem = 4GB
work_mem = 64MB
max_connections = 200

# Restart PostgreSQL
sudo systemctl restart postgresql
```

## Uninstallation

To remove the RAG platform:

```bash
# Stop all services
sudo ./manage-services.sh stop

# Disable auto-start
sudo ./manage-services.sh disable

# Remove systemd service files
sudo rm /etc/systemd/system/{weaviate,rag-api,web-api,web-ui}.service
sudo systemctl daemon-reload

# Remove Weaviate
sudo rm -rf /opt/weaviate

# Remove data (⚠️ destroys all data)
rm -rf /home/rag/src/rag/weaviate_data
sudo -u postgres psql -c "DROP DATABASE ragdb;"
sudo -u postgres psql -c "DROP USER raguser;"

# Remove packages (optional)
sudo apt-get remove --purge postgresql-16 redis-server nginx
sudo apt-get autoremove
```

## Updating the Platform

### Update Application Code

```bash
cd /home/rag/src/rag
git pull origin master

# Update Python dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Update Node dependencies
cd services/web-api && npm update
cd ../web-ui && npm update

# Restart services
sudo ./manage-services.sh restart
```

### Update Weaviate

```bash
# Download new version
wget https://github.com/weaviate/weaviate/releases/download/vX.X.X/weaviate-vX.X.X-linux-amd64.tar.gz

# Stop service
sudo systemctl stop weaviate

# Backup data
cp -r /home/rag/src/rag/weaviate_data /home/rag/src/rag/weaviate_data.backup

# Replace binary
tar -xzf weaviate-vX.X.X-linux-amd64.tar.gz
sudo mv weaviate /opt/weaviate/

# Start service
sudo systemctl start weaviate
```

## Additional Resources

- **RAG CLI Documentation**: See `/home/rag/src/rag/docs/`
- **Weaviate Documentation**: https://weaviate.io/developers/weaviate
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/16/
- **Nginx Documentation**: https://nginx.org/en/docs/

## Support and Feedback

For issues specific to this installation:
1. Check service logs: `sudo ./manage-services.sh logs <service>`
2. Verify health checks: `sudo ./manage-services.sh health`
3. Review configuration files in `/etc/systemd/system/`

---

**Installation completed successfully!** 🎉

Access your RAG platform at: `http://localhost` or `http://your-server-ip`
