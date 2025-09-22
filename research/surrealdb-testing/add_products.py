#!/usr/bin/env python3
"""
Script to add sample products to the SurrealDB companies
"""

from surrealdb import Surreal

def add_products():
    print("=== Adding Products to SurrealDB ===")

    try:
        with Surreal("http://localhost:8000") as db:
            db.signin({"username": "admin", "password": "ProductionAdminPass2024"})
            db.use("production", "main")

            # First get existing companies
            companies = db.select("companies")
            print(f"Found {len(companies)} companies")

            if not companies:
                print("No companies found! Please run the app first to create companies.")
                return

            # Create products for each company
            products_data = []

            for company in companies:
                company_id = company['id']
                company_name = company['name']
                industry = company['industry']

                print(f"Adding products for {company_name} ({industry})")

                if industry == "Technology":
                    products_data.extend([
                        {
                            "name": "CloudSync Pro",
                            "company": company_id,
                            "category": "Software",
                            "price": 299.99,
                            "active": True
                        },
                        {
                            "name": "AI Analytics Dashboard",
                            "company": company_id,
                            "category": "Analytics",
                            "price": 599.99,
                            "active": True
                        }
                    ])
                elif industry == "Renewable Energy":
                    products_data.extend([
                        {
                            "name": "Solar Panel X1",
                            "company": company_id,
                            "category": "Hardware",
                            "price": 1299.99,
                            "active": True
                        },
                        {
                            "name": "Wind Turbine Controller",
                            "company": company_id,
                            "category": "Hardware",
                            "price": 2499.99,
                            "active": True
                        }
                    ])
                elif industry == "Financial Technology":
                    products_data.extend([
                        {
                            "name": "CryptoWallet Security",
                            "company": company_id,
                            "category": "Security",
                            "price": 49.99,
                            "active": True
                        },
                        {
                            "name": "Payment Gateway API",
                            "company": company_id,
                            "category": "Software",
                            "price": 199.99,
                            "active": True
                        }
                    ])

            # Insert all products
            print(f"\nInserting {len(products_data)} products...")
            for product in products_data:
                result = db.create("products", product)
                print(f"Created product: {result['name']} for {result['company']}")

            print(f"\n✅ Successfully added {len(products_data)} products!")

            # Verify by counting products
            product_count = db.query("SELECT count() FROM products GROUP ALL;")
            print(f"Total products in database: {product_count}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    add_products()