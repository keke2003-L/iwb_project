Distributed Online Shopping Database Application
Overview
A Flask-based web application with a distributed MySQL database for managing an online store, including product listings, shopping carts, user accounts, and sales reports.
Prerequisites

Python 3.8+
MySQL Server 8.0+
pip for installing dependencies

MySQL Setup

Install MySQL Server and ensure it’s running.
Create two databases:CREATE DATABASE store_db;
CREATE DATABASE customers_db;


Create a MySQL user and grant privileges:CREATE USER 'your_username'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON store_db.* TO 'your_username'@'localhost';
GRANT ALL PRIVILEGES ON customers_db.* TO 'your_username'@'localhost';
FLUSH PRIVILEGES;


Update app.py with your MySQL credentials (your_username, your_password).

Project Setup

Clone the repository:git clone <repository-url>
cd project_root


Install dependencies:pip install -r requirements.txt


Ensure the templates/ folder is in the project root.
Run the application:python app.py


Open http://127.0.0.1:5000 in a browser.

Features

Distributed database across two MySQL instances.
User registration/login with secure password hashing.
Product browsing, cart management, and checkout.
Admin panel for product management.
Sales reports by category.

Notes

Use admin@example.com as the admin email for testing admin features.
MySQL databases are initialized automatically on first run.
Ensure MySQL server is running before starting the application.

Troubleshooting

If you encounter connection errors, verify MySQL credentials and ensure the server is running.
Check that the mysql-connector-python library is installed correctly.

