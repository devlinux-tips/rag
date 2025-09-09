#!/bin/bash
echo "🗄️  Starting SurrealDB..."
surreal start --log trace --user root --pass root file://data/surrealdb
