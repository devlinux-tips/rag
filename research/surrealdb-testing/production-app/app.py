#!/usr/bin/env python3
"""
Production SurrealDB Query Application
Simple Flask app demonstrating production-ready SurrealDB integration
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify
from surrealdb import Surreal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Production configuration from environment
SURREALDB_URL = os.getenv('SURREALDB_URL', 'http://localhost:8000')
SURREALDB_USER = os.getenv('SURREALDB_USER', 'admin')
SURREALDB_PASS = os.getenv('SURREALDB_PASS', 'ProductionAdminPass2024')

class ProductionSurrealDB:
    """Production-ready SurrealDB client using Python SDK"""

    def __init__(self):
        self.base_url = SURREALDB_URL
        self.username = SURREALDB_USER
        self.password = SURREALDB_PASS
        self.namespace = "production"
        self.database = "main"

        # Initialize the database schema and data
        self.setup_production_data()

    def _get_connection(self):
        """Get a new SurrealDB connection"""
        db = Surreal(self.base_url)
        db.signin({"username": self.username, "password": self.password})
        db.use(self.namespace, self.database)
        return db

    def setup_production_data(self):
        """Setup production database schema and sample data"""
        try:
            with self._get_connection() as db:
                # First create namespace and database
                db.query("""
                    DEFINE NAMESPACE production;
                    USE NS production;
                    DEFINE DATABASE main;
                """)

                # Reconnect with namespace/database context
                db.use(self.namespace, self.database)

                # Create companies table schema
                db.query("""
                    DEFINE TABLE companies SCHEMAFULL;
                    DEFINE FIELD name ON companies TYPE string ASSERT $value != NONE;
                    DEFINE FIELD industry ON companies TYPE string ASSERT $value != NONE;
                    DEFINE FIELD founded_year ON companies TYPE number;
                    DEFINE FIELD employees ON companies TYPE number;
                    DEFINE FIELD revenue_millions ON companies TYPE number;
                    DEFINE FIELD created_at ON companies TYPE datetime DEFAULT time::now();
                """)

                # Create products table schema
                db.query("""
                    DEFINE TABLE products SCHEMAFULL;
                    DEFINE FIELD name ON products TYPE string ASSERT $value != NONE;
                    DEFINE FIELD company ON products TYPE record<companies>;
                    DEFINE FIELD category ON products TYPE string ASSERT $value != NONE;
                    DEFINE FIELD price ON products TYPE number;
                    DEFINE FIELD active ON products TYPE bool DEFAULT true;
                    DEFINE FIELD created_at ON products TYPE datetime DEFAULT time::now();
                """)

                # Check if companies already exist to avoid duplicates
                existing_companies = db.select("companies")
                if not existing_companies:
                    # Insert sample companies
                    companies_data = [
                        {
                            "name": "TechCorp Industries",
                            "industry": "Technology",
                            "founded_year": 2015,
                            "employees": 2500,
                            "revenue_millions": 450
                        },
                        {
                            "name": "Green Energy Solutions",
                            "industry": "Renewable Energy",
                            "founded_year": 2018,
                            "employees": 850,
                            "revenue_millions": 125
                        },
                        {
                            "name": "FinTech Plus",
                            "industry": "Financial Technology",
                            "founded_year": 2020,
                            "employees": 320,
                            "revenue_millions": 75
                        }
                    ]

                    for company_data in companies_data:
                        db.create("companies", company_data)

                    logger.info("Production database setup completed successfully")
                else:
                    logger.info("Database already populated, skipping data insertion")

        except Exception as e:
            logger.error(f"Setup error: {e}")

    def get_companies(self):
        """Get all companies ordered by revenue"""
        try:
            with self._get_connection() as db:
                companies = db.query("""
                    SELECT
                        id,
                        name,
                        industry,
                        founded_year,
                        employees,
                        revenue_millions
                    FROM companies
                    ORDER BY revenue_millions DESC;
                """)

                # Convert RecordID objects to strings for JSON serialization
                result = []
                for company in companies:
                    company_dict = dict(company)
                    if 'id' in company_dict:
                        company_dict['id'] = str(company_dict['id'])
                    if 'created_at' in company_dict:
                        company_dict['created_at'] = company_dict['created_at'].isoformat()
                    result.append(company_dict)

                return result
        except Exception as e:
            logger.error(f"Error getting companies: {e}")
            return []

    def get_company_by_id(self, company_id):
        """Get a specific company by ID"""
        try:
            with self._get_connection() as db:
                # Use the select method with specific record ID
                company = db.select(company_id)

                if company:
                    # Convert RecordID objects to strings for JSON serialization
                    company_dict = dict(company)
                    if 'id' in company_dict:
                        company_dict['id'] = str(company_dict['id'])
                    if 'created_at' in company_dict:
                        company_dict['created_at'] = company_dict['created_at'].isoformat()
                    return company_dict
                else:
                    return None
        except Exception as e:
            logger.error(f"Error getting company {company_id}: {e}")
            return None

    def get_products_by_company(self, company_id):
        """Get products for a specific company"""
        try:
            with self._get_connection() as db:
                products = db.query(f"""
                    SELECT
                        name,
                        category,
                        price,
                        active,
                        company.name AS company_name
                    FROM products
                    WHERE company = {company_id}
                    ORDER BY price DESC;
                """)

                # Convert any RecordID objects to strings for JSON serialization
                result = []
                for product in products:
                    product_dict = dict(product)
                    if 'id' in product_dict:
                        product_dict['id'] = str(product_dict['id'])
                    if 'created_at' in product_dict:
                        product_dict['created_at'] = product_dict['created_at'].isoformat()
                    result.append(product_dict)

                return result
        except Exception as e:
            logger.error(f"Error getting products for company {company_id}: {e}")
            return []

    def get_industry_stats(self):
        """Get statistics by industry"""
        try:
            with self._get_connection() as db:
                stats = db.query("""
                    SELECT
                        industry,
                        count() AS company_count,
                        math::sum(employees) AS total_employees,
                        math::sum(revenue_millions) AS total_revenue
                    FROM companies
                    GROUP BY industry
                    ORDER BY total_revenue DESC;
                """)

                # Convert any RecordID objects to strings for JSON serialization
                result = []
                for stat in stats:
                    stat_dict = dict(stat)
                    result.append(stat_dict)

                return result
        except Exception as e:
            logger.error(f"Error getting industry stats: {e}")
            return []

# Initialize database connection
db = ProductionSurrealDB()

@app.route('/')
def index():
    """Main dashboard showing companies and statistics"""
    try:
        companies = db.get_companies()
        industry_stats = db.get_industry_stats()

        return render_template('index.html',
                             companies=companies,
                             industry_stats=industry_stats,
                             total_companies=len(companies))
    except Exception as e:
        logger.error(f"Index route error: {e}")
        return render_template('index.html',
                             companies=[],
                             industry_stats=[],
                             total_companies=0,
                             error=str(e))

@app.route('/api/companies')
def api_companies():
    """API endpoint for companies data"""
    try:
        companies = db.get_companies()
        return jsonify({
            "status": "success",
            "data": companies,
            "count": len(companies)
        })
    except Exception as e:
        logger.error(f"API companies error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/products/<company_id>')
def api_products(company_id):
    """API endpoint for products by company"""
    try:
        products = db.get_products_by_company(f"companies:{company_id}")
        return jsonify({
            "status": "success",
            "data": products,
            "count": len(products)
        })
    except Exception as e:
        logger.error(f"API products error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for industry statistics"""
    try:
        stats = db.get_industry_stats()
        return jsonify({
            "status": "success",
            "data": stats,
            "count": len(stats)
        })
    except Exception as e:
        logger.error(f"API stats error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Test database connection using Python SDK
        with db._get_connection() as test_db:
            result = test_db.query("INFO FOR DB;")
            db_status = "connected" if result is not None else "disconnected"

        return jsonify({
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.now().isoformat(),
            "surrealdb_url": SURREALDB_URL
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    logger.info("Starting Production SurrealDB Query Application")
    logger.info(f"SurrealDB URL: {SURREALDB_URL}")

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)