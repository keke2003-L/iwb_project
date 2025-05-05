import mysql.connector
from mysql.connector import pooling
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, SelectField, TextAreaField, FloatField, SubmitField, IntegerField, HiddenField
from wtforms.validators import DataRequired, Email, NumberRange, Length, Regexp, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import uuid
import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import difflib
import json
import logging
from functools import wraps
from werkzeug.utils import secure_filename

# Check for email_validator
try:
    import email_validator
except ImportError:
    logging.error("email_validator package is missing. Please install it with 'pip install email_validator'.")
    # Optionally, you can exit or raise a more specific error
    # In a real application, you might want to handle this more gracefully
    pass  # Allow the app to start, but log the critical error

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('FLASK_ENV') == 'development' else logging.INFO,  # Log DEBUG in dev, INFO in production
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- FIX ---
# Set the secret key *before* initializing CSRFProtect
# Use the hardcoded string for testing as discussed
app.secret_key = 'a-very-secure-and-random-string-for-testing'
# In production, use an environment variable:
# app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-fallback-secret-key-secure-this-in-production')
# Make sure 'your-fallback-secret-key-secure-this-in-production' is replaced with a real secret key for production.

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Session timeout
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV', 'development') != 'development'  # Disable secure cookies in development
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JS access to cookies
# 'Strict' is even better for CSRF, but 'Lax' is a good balance
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# CSRF protection - This line should be *after* app.secret_key is set
csrf = CSRFProtect(app)

# Ensure upload and charts folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

# Valid categories for IWB products
VALID_CATEGORIES = ['RAM', 'Hard Drives', 'Motherboard Components']
VALID_ROLES = ['customer', 'sales', 'finance', 'developer', 'investor']

# MySQL connection pool configurations
# Load connection details from environment variables for better security
TENANT1_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    # --- SECURITY WARNING ---
    # **Highly recommend using environment variables for passwords**
    'password': os.environ.get('DB_PASSWORD', '59482807'),
    'database': 'iwb_public'
}

TENANT2_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    # --- SECURITY WARNING ---
    # **Highly recommend using environment variables for passwords**
    'password': os.environ.get('DB_PASSWORD', '59482807'),
    'database': 'iwb_private'
}

# Initialize connection pools - Moved inside __main__ block for better startup control
tenant1_pool = None
tenant2_pool = None

# Helper functions
def get_tenant1():
    """Acquires a connection from the tenant1 pool and attempts immediate rollback."""
    global tenant1_pool  # Need global to access the pool variable initialized elsewhere
    if tenant1_pool is None:
        logger.critical("tenant1_pool is not initialized.")
        # This indicates a critical startup failure, re-raise or handle appropriately
        raise mysql.connector.Error("Database connection pool not initialized.")
    conn = None  # Initialize conn before pool.get_connection
    try:
        conn = tenant1_pool.get_connection()
        logger.debug("Acquired connection from tenant1_pool")
        # --- Added: Attempt rollback immediately after acquiring ---
        # This helps ensure the connection is in a clean state from previous use.
        try:
            if conn.is_connected():  # Check before attempting rollback
                conn.rollback()
                logger.debug("Successfully attempted rollback immediately after acquiring connection from tenant1_pool.")
        except mysql.connector.Error as rollback_e:
            logger.warning(f"Error during immediate rollback after acquiring connection from tenant1_pool: {rollback_e}")
        # --- End Added ---
        return conn
    except mysql.connector.Error as e:
        logger.error(f"Error getting connection from tenant1_pool: {e}")
        # Ensure conn is closed if it was partially acquired before failing
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Safeguard rollback
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback on acquisition failure from tenant1_pool: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after acquisition failure from tenant1_pool.")
        # Re-raise the exception so the error handler can catch it
        raise

def get_tenant2():
    """Acquires a connection from the tenant2 pool and attempts immediate rollback."""
    global tenant2_pool  # Need global to access the pool variable initialized elsewhere
    if tenant2_pool is None:
        logger.critical("tenant2_pool is not initialized.")
        # This indicates a critical startup failure, re-raise or handle appropriately
        raise mysql.connector.Error("Database connection pool not initialized.")
    conn = None  # Initialize conn before pool.get_connection
    try:
        conn = tenant2_pool.get_connection()
        logger.debug("Acquired connection from tenant2_pool")
        # --- Added: Attempt rollback immediately after acquiring ---
        # This helps ensure the connection is in a clean state from previous use.
        try:
            if conn.is_connected():  # Check before attempting rollback
                conn.rollback()
                logger.debug("Successfully attempted rollback immediately after acquiring connection from tenant2_pool.")
        except mysql.connector.Error as rollback_e:
            logger.warning(f"Error during immediate rollback after acquiring connection from tenant2_pool: {rollback_e}")
        # --- End Added ---
        return conn
    except mysql.connector.Error as e:
        logger.error(f"Error getting connection from tenant2_pool: {e}")
        # Ensure conn is closed if it was partially acquired before failing
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Safeguard rollback
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback on acquisition failure from tenant2_pool: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after acquisition failure from tenant2_pool.")
        # Re-raise the exception so the error handler can catch it
        raise

# Database initialization
# iwb_public: products, orders, order_items, cart
def init_tenant1():
    conn = None
    try:
        conn = get_tenant1()  # Acquire connection from the pool
        c = conn.cursor()
        # Using VARCHAR(36) for UUIDs is standard
        c.execute('''CREATE TABLE IF NOT EXISTS products
                     (productID VARCHAR(36) PRIMARY KEY, name VARCHAR(255), description TEXT, price DECIMAL(10,2), stock INT, category VARCHAR(100), image_path VARCHAR(255))''')
        c.execute('''CREATE TABLE IF NOT EXISTS orders
                     (orderID VARCHAR(36) PRIMARY KEY, customerID VARCHAR(36), date DATETIME, total DECIMAL(10,2), status VARCHAR(50))''')
        c.execute('''CREATE TABLE IF NOT EXISTS order_items
                     (orderID VARCHAR(36), productID VARCHAR(36), quantity INT, unit_price DECIMAL(10,2),
                      PRIMARY KEY (orderID, productID),
                      FOREIGN KEY (orderID) REFERENCES orders(orderID) ON DELETE CASCADE,
                      FOREIGN KEY (productID) REFERENCES products(productID) ON DELETE CASCADE)''')  # Add foreign key constraints
        c.execute('''CREATE TABLE IF NOT EXISTS cart
                     (customerID VARCHAR(36), productID VARCHAR(36), quantity INT,
                      PRIMARY KEY (customerID, productID),
                      FOREIGN KEY (productID) REFERENCES products(productID) ON DELETE CASCADE)''')  # Add foreign key constraint (customerID FK would need a users table in public or cross-db FK which is complex)

        # Insert sample products (using INSERT IGNORE to avoid errors if they exist)
        sample_products = [
            ('1', '8GB DDR4 RAM', 'High-performance DDR4 RAM module', 50.00, 100, 'RAM', 'static/images/ram.jpg'),
            ('2', '1TB SSD', 'Fast and reliable solid-state drive', 100.00, 50, 'Hard Drives', 'static/images/ssd.jpg'),
            ('3', 'Intel Core i7 CPU', 'Powerful processor for gaming and multitasking', 300.00, 30, 'Motherboard Components', 'static/images/cpu.jpg'),
            ('4', '16GB DDR4 RAM', 'High-capacity DDR4 RAM module', 80.00, 80, 'RAM', 'static/images/ram_16gb.jpg'),
            ('5', '2TB HDD', 'Large-capacity hard drive', 60.00, 70, 'Hard Drives', 'static/images/hdd.jpg'),
            ('6', 'AMD Ryzen 5 CPU', 'Efficient processor for everyday use', 200.00, 40, 'Motherboard Components', 'static/images/cpu_amd.jpg')
        ]
        # Using INSERT IGNORE is safer than ON DUPLICATE KEY UPDATE for simple inserts
        c.executemany('''INSERT IGNORE INTO products (productID, name, description, price, stock, category, image_path)
                         VALUES (%s, %s, %s, %s, %s, %s, %s)''', sample_products)

        conn.commit()
        logger.info("Database iwb_public initialized successfully.")
    except mysql.connector.Error as e:
        logger.error(f"Error initializing iwb_public: {e}")
        # Re-raise the exception to signal initialization failure
        raise
    finally:
        # --- FIX ---
        # Ensure connection is closed if it was successfully acquired
        if conn and conn.is_connected():
            # Rollback any pending transaction before closing/returning to pool.
            try:
                conn.rollback()  # Safe rollback after commit or on error
                logger.debug("Rolled back any pending transaction during init_tenant1 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in init_tenant1 finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed in init_tenant1 finally block.")

# iwb_private: users, inquiries, income_statements, files
def init_tenant2():
    conn = None
    try:
        conn = get_tenant2()  # Acquire connection from the pool
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (userID VARCHAR(36) PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE, password VARCHAR(255),
                      role VARCHAR(20))''')
        c.execute('''CREATE TABLE IF NOT EXISTS inquiries
                     (inquiryID VARCHAR(36) PRIMARY KEY, name VARCHAR(255), email VARCHAR(255), message TEXT, submitted_at DATETIME,
                      status VARCHAR(20) DEFAULT 'pending', response TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS income_statements
                     (statementID VARCHAR(36) PRIMARY KEY, month VARCHAR(7), revenue DECIMAL(10,2), expenses DECIMAL(10,2),
                      net_income DECIMAL(10,2), chart_path VARCHAR(255))''')
        c.execute('''CREATE TABLE IF NOT EXISTS files
                     (fileID VARCHAR(36) PRIMARY KEY, name VARCHAR(255), path VARCHAR(255), uploaded_by VARCHAR(36), uploaded_at DATETIME,
                      FOREIGN KEY (uploaded_by) REFERENCES users(userID) ON DELETE SET NULL)''')  # Add foreign key constraint

        conn.commit()
        logger.info("Database iwb_private initialized successfully.")
    except mysql.connector.Error as e:
        logger.error(f"Error initializing iwb_private: {e}")
        # Re-raise the exception to signal initialization failure
        raise
    finally:
        # --- FIX ---
        # Ensure connection is closed if it was successfully acquired
        if conn and conn.is_connected():
            # Rollback any pending transaction before closing/returning to pool.
            try:
                conn.rollback()  # Safe rollback after commit or on error
                logger.debug("Rolled back any pending transaction during init_tenant2 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in init_tenant2 finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed in init_tenant2 finally block.")

# Verify required tables exist
def verify_tables():
    conn1 = None
    conn2 = None
    try:
        conn1 = get_tenant1()
        conn2 = get_tenant2()
        c1 = conn1.cursor()
        c2 = conn2.cursor()
        c1.execute("SHOW TABLES")
        tables1 = {row[0] for row in c1.fetchall()}  # Use a set for easier comparison
        c2.execute("SHOW TABLES")
        tables2 = {row[0] for row in c2.fetchall()}  # Use a set for easier comparison
        required_public = {'products', 'orders', 'order_items', 'cart'}
        required_private = {'users', 'inquiries', 'income_statements', 'files'}
        if not required_public.issubset(tables1):
            missing = required_public - tables1
            logger.error(f"Missing tables in iwb_public: {missing}")
            raise Exception(f"Missing required tables in iwb_public: {missing}")
        if not required_private.issubset(tables2):
            missing = required_private - tables2
            logger.error(f"Missing tables in iwb_private: {missing}")
            raise Exception(f"Missing required tables in iwb_private: {missing}")
        logger.info("All required tables verified.")
    except mysql.connector.Error as e:
        logger.error(f"Table verification failed: {e}")
        raise  # Re-raise to stop execution if verification fails
    except Exception as e:  # Catch the specific Exception raised above
        logger.error(f"Table verification failed: {e}")
        raise  # Re-raise to stop execution if verification fails
    finally:
        # --- FIX ---
        # Ensure connections are closed if successfully acquired
        if conn1 and conn1.is_connected():
            try:
                conn1.rollback()  # Rollback in case verification started a transaction (unlikely but safe)
                logger.debug("Rolled back pending transaction during verify_tables conn1 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in verify_tables conn1 finally block: {rollback_e}")
            conn1.close()
            logger.debug("Connection conn1 closed in verify_tables finally block.")
        if conn2 and conn2.is_connected():
            try:
                conn2.rollback()  # Rollback in case verification started a transaction
                logger.debug("Rolled back pending transaction during verify_tables conn2 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in verify_tables conn2 finally block: {rollback_e}")
            conn2.close()
            logger.debug("Connection conn2 closed in verify_tables finally block.")

# WTForms for validation
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=255)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8),
        Regexp(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*?&]{8,}$', message="Password must contain at least one letter and one number")
    ])
    role = SelectField('Role', choices=[(role, role.capitalize()) for role in VALID_ROLES], validators=[DataRequired()])
    submit = SubmitField('Register')

    # Custom validator to check role limits
    def validate_role(self, field):
        conn = None
        try:
            conn = get_tenant2()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM users WHERE role = %s", (field.data,))
            count = c.fetchone()[0]
            # --- FIX ---
            # Close connection after fetching count
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback after read operation
                    logger.debug("Rolled back pending transaction during validate_role fetch finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in validate_role fetch finally block: {rollback_e}")
                conn.close()
                conn = None  # Set to None

            if (field.data in ['sales', 'finance', 'developer'] and count >= 3) or (field.data == 'investor' and count >= 10):
                raise ValidationError(f'Maximum {field.data} accounts reached.')
        except mysql.connector.Error as e:
            logger.error(f"Role validation database error: {e}")
            # --- FIX ---
            # Rollback on error
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback on error
                    logger.debug("Rolled back pending transaction during validate_role error finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in validate_role error finally block: {rollback_e}")
                conn.close()
                conn = None  # Set to None
            # Raise a specific error for validation to catch
            raise ValidationError("Database error during role validation.")
        except Exception as e:
            logger.error(f"Unexpected error during role validation: {e}", exc_info=True)
            # --- FIX ---
            # Rollback on unexpected error
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback on error
                    logger.debug("Rolled back pending transaction during validate_role unexpected error finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in validate_role unexpected error finally block: {rollback_e}")
                conn.close()
                conn = None  # Set to None
            raise ValidationError("An unexpected error occurred during role validation.")
        finally:
            # This finally block is a safeguard if an exception occurred *before*
            # the explicit conn.close() after fetching count.
            if conn and conn.is_connected():
                logger.warning(f"Connection was still open in validate_role final finally block. Closing now.")
                try:
                    conn.rollback()  # Rollback as a safeguard
                    logger.debug("Rolled back pending transaction in validate_role final finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in validate_role final finally block: {rollback_e}")
                conn.close()
                logger.debug("Database connection closed in validate_role final finally block.")

class InquiryForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=255)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    message = TextAreaField('Message', validators=[DataRequired(), Length(max=1000)])
    submit = SubmitField('Submit Inquiry')

class IncomeStatementForm(FlaskForm):
    month = StringField('Month (YYYY-MM)', validators=[DataRequired(), Regexp(r'^\d{4}-\d{2}$', message="Format: YYYY-MM")])
    revenue = FloatField('Revenue', validators=[DataRequired(), NumberRange(min=0)])
    expenses = FloatField('Expenses', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Create Statement')

# New form for adding to cart
class AddToCartForm(FlaskForm):
    quantity = IntegerField('Quantity', validators=[DataRequired(), NumberRange(min=1, message="Quantity must be at least 1")])
    submit = SubmitField('Add to Cart')

# New form for checkout
class CheckoutForm(FlaskForm):
    card_number = StringField('Card Number', validators=[DataRequired(), Length(min=16, max=16, message="Invalid card number")])  # Basic validation
    expiry_date = StringField('Expiry Date (MM/YY)', validators=[DataRequired(), Regexp(r'^\d{2}\/\d{2}$', message="Format: MM/YY")])
    cvv = StringField('CVV', validators=[DataRequired(), Length(min=3, max=4, message="Invalid CVV")])
    submit = SubmitField('Place Order')

# New form for sales response
class SalesResponseForm(FlaskForm):
    inquiryID = HiddenField('Inquiry ID', validators=[DataRequired()])
    response = TextAreaField('Response', validators=[DataRequired(), Length(max=1000)])
    submit = SubmitField('Submit Response')

# Helper functions (continued)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_income_chart(statement):
    """Generates and saves an income statement chart."""
    months = [statement['month']]
    revenue = [statement['revenue']]
    expenses = [statement['expenses']]
    net_income = [statement['net_income']]

    x = np.arange(len(months))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width, revenue, width, label='Revenue', color='green')
    ax.bar(x, expenses, width, label='Expenses', color='red')
    ax.bar(x + width, net_income, width, label='Net Income', color='blue')

    ax.set_xlabel('Month')
    ax.set_ylabel('Amount')  # Use 'Amount' instead of 'Amount (M)' unless values are in millions
    ax.set_title(f'Income Statement - {statement["month"]}')
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()
    ax.grid(axis='y')  # Only show horizontal grid lines

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)  # Close the figure to free up memory
    chart_path = f"static/charts/income_statement_{statement['statementID']}.png"
    try:
        with open(chart_path, 'wb') as f:
            f.write(buffer.getvalue())
        logger.debug(f"Chart saved to {chart_path}")
        return chart_path
    except IOError as e:
        logger.error(f"Error saving chart to {chart_path}: {e}")
        return None

def auto_respond_query(message):
    """Provides basic auto-responses based on keywords."""
    responses = {
        "price": "Our prices vary by component. Please check our product listings for details.",
        "recycling process": "We recycle RAM, Hard Drives, and Motherboard Components using eco-friendly methods.",
        "delivery": "Delivery times depend on your location. Contact us for a quote."
    }
    # Use a more robust keyword matching or NLP if needed
    message_lower = message.lower()
    for key, response in responses.items():
        if key.lower() in message_lower:  # Simple substring match
            return response
        # You can keep the difflib logic if preferred, but substring is simpler
        # if difflib.SequenceMatcher(None, key.lower(), message.lower()).ratio() > 0.7:
        #     return response
    return None

# Role-based access decorator
def require_role(role):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to access this page.', 'warning')  # Changed to warning
                return redirect(url_for('login'))
            conn = None  # Initialize conn to None
            try:
                conn = get_tenant2()
                c = conn.cursor(dictionary=True)
                # Use LIMIT 1 as email is unique
                c.execute("SELECT role FROM users WHERE userID = %s LIMIT 1", (session['user_id'],))
                user = c.fetchone()

                # --- FIX ---
                # Close connection immediately after fetching user data
                if conn and conn.is_connected():
                    try:
                        conn.rollback()  # Rollback after read operation
                        logger.debug("Rolled back pending transaction during require_role fetch finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback in require_role fetch finally block: {rollback_e}")
                    conn.close()
                    conn = None  # Set to None so the final finally doesn't try to close it again

                if not user:
                    flash('User not found. Please log in again.', 'error')
                    session.clear()  # Clear session if user is not found
                    return redirect(url_for('login'))

                if user['role'] != role:
                    flash(f'Access denied. You must be a {role} to view this page.', 'danger')  # Changed to danger
                    # Redirect to a sensible default like dashboard or index
                    return redirect(url_for('dashboard' if 'user_role' in session else 'index'))

                # Update session role just in case (though it should be set on login)
                session['user_role'] = user['role']
                return f(*args, **kwargs)
            except mysql.connector.Error as e:
                logger.error(f"Role check database error for user {session['user_id']}: {e}")
                flash(f"A database error occurred during role check.", 'error')
                # --- FIX ---
                # Ensure connection is closed on error path
                if conn and conn.is_connected():
                    try:
                        conn.rollback()  # Rollback on error
                        logger.debug("Rolled back pending transaction during require_role error finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback in require_role error finally block: {rollback_e}")
                    conn.close()
                    conn = None  # Set to None
                # Redirect to an error page or dashboard
                return redirect(url_for('dashboard' if 'user_role' in session else 'index'))
            except Exception as e:
                logger.error(f"Unexpected error during role check for user {session['user_id']}: {e}", exc_info=True)
                flash(f"An unexpected error occurred during role check.", 'error')
                # --- FIX ---
                # Ensure connection is closed on unexpected error path
                if conn and conn.is_connected():
                    try:
                        conn.rollback()  # Rollback on error
                        logger.debug("Rolled back pending transaction during require_role unexpected error finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback in require_role unexpected error finally block: {rollback_e}")
                    conn.close()
                    conn = None  # Set to None
                return redirect(url_for('dashboard' if 'user_role' in session else 'index'))
            finally:
                # This finally block is a safeguard if an exception occurred *before*
                # the explicit conn.close() after fetching user data.
                if conn and conn.is_connected():
                    logger.warning(f"Connection was still open in require_role final finally block. Closing now.")
                    try:
                        conn.rollback()  # Rollback as a safeguard
                        logger.debug("Rolled back pending transaction in require_role final finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback in require_role final finally block: {rollback_e}")
                    conn.close()
                    logger.debug("Database connection closed in require_role final finally block.")

        return wrapper
    return decorator

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 error: {request.url}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}", exc_info=True)  # Log traceback
    # You might want to render a more user-friendly error page in production
    return render_template('500.html'), 500

# Generic database error handler
@app.errorhandler(mysql.connector.Error)
def handle_db_error(e):
    logger.error(f"Database error occurred: {e}", exc_info=True)
    flash("A database error occurred. Please try again later.", 'error')
    # Redirect to a safe page, like the index or an error page
    return redirect(url_for('index'))

# Routes
@app.route('/')
def index():
    category = request.args.get('category', '')
    conn = None  # Initialize conn to None
    products = []
    try:
        conn = get_tenant1()
        c = conn.cursor(dictionary=True)
        if category and category in VALID_CATEGORIES:
            c.execute("SELECT * FROM products WHERE category = %s", (category,))
        else:
            # Fetch all if category is not valid or not provided
            c.execute("SELECT * FROM products")
        products = c.fetchall()
        logger.debug(f"Fetched {len(products)} products for category '{category}'")
    except mysql.connector.Error as e:
        logger.error(f"Index route database error: {e}")
        flash(f"Failed to load products.", 'error')  # Generic user message
    except Exception as e:
        logger.error(f"Unexpected error in index route: {e}", exc_info=True)
        flash(f"An unexpected error occurred.", 'error')
    finally:
        # --- FIX ---
        # Ensure connection is closed if successfully acquired
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation (safe)
                logger.debug("Rolled back pending transaction during index finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in index finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed in index finally block.")

    return render_template('index.html', products=products, selected_category=category, categories=VALID_CATEGORIES)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        conn = None  # Initialize conn to None
        try:
            logger.debug(f"Attempting login for email: {email}")
            conn = get_tenant2()
            c = conn.cursor(dictionary=True)
            # Use LIMIT 1 as email is unique
            c.execute("SELECT userID, name, password, role FROM users WHERE email = %s LIMIT 1", (email,))
            user = c.fetchone()
            # logger.debug(f"User query result: {'User found' if user else 'User not found'}")

            # --- FIX ---
            # Close connection immediately after fetching user data
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback after read operation
                    logger.debug("Rolled back pending transaction during login fetch finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in login fetch finally block: {rollback_e}")
                conn.close()
                conn = None  # Set to None so the final finally doesn't try to close it again

            if user and check_password_hash(user['password'], password):
                session.permanent = True
                session['user_id'] = user['userID']
                session['user_name'] = user['name']
                session['user_role'] = user['role']
                logger.info(f"Login successful for user: {email} (Role: {user['role']})")
                flash('Login successful.', 'success')
                return redirect(url_for('dashboard'))
            else:
                logger.warning(f"Invalid credentials attempt for email: {email}")
                flash('Invalid email or password.', 'error')
        except mysql.connector.Error as e:
            logger.error(f"Login database failed for {email}: {e}")
            flash('An error occurred during login. Please try again.', 'error')  # Generic user message
            # --- FIX ---
            # Ensure connection is closed on error path
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback on error
                    logger.debug("Rolled back pending transaction during login error finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in login error finally block: {rollback_e}")
                conn.close()
                conn = None  # Set to None
        except Exception as e:
            logger.error(f"Unexpected error during login for {email}: {e}", exc_info=True)
            flash('An unexpected error occurred during login. Please try again.', 'error')
            # --- FIX ---
            # Ensure connection is closed on unexpected error path
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback on error
                    logger.debug("Rolled back pending transaction during login unexpected error finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in login unexpected error finally block: {rollback_e}")
                conn.close()
                conn = None  # Set to None
        finally:
            # This finally block is a safeguard if an exception occurred *before*
            # the explicit conn.close() after fetching user data.
            if conn and conn.is_connected():
                logger.warning(f"Connection was still open in login final finally block. Closing now.")
                try:
                    conn.rollback()  # Rollback as a safeguard
                    logger.debug("Rolled back pending transaction during login final finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in login final finally block: {rollback_e}")
                conn.close()
                logger.debug("Database connection closed in login final finally block.")

    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = generate_password_hash(form.password.data)
        role = form.role.data
        userID = str(uuid.uuid4())

        conn = None  # Initialize conn to None
        try:
            conn = get_tenant2()
            c = conn.cursor()
            # The role limit check is now handled by the form validator
            c.execute("INSERT INTO users (userID, name, email, password, role) VALUES (%s, %s, %s, %s, %s)",
                      (userID, name, email, password, role))
            conn.commit()
            logger.info(f"User registered: {email} (Role: {role})")
            flash('Registration successful! Please log in.', 'success')
            # --- FIX ---
            # Close connection after successful commit before redirect
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Commit was successful, rollback won't do anything but is safe
                    logger.debug("Rolled back pending transaction during register successful commit finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in register successful commit finally block: {rollback_e}")
                conn.close()
                conn = None  # Set to None so the final finally doesn't try to close it again

            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:  # Handle duplicate email specifically
            flash('Email address already registered.', 'error')
            logger.warning(f"Registration failed due to duplicate email: {email}")
            # --- FIX ---
            # Rollback transaction on error
            if conn and conn.is_connected():
                try:
                    conn.rollback()
                    logger.debug("Rolled back transaction due to IntegrityError during register.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback after IntegrityError during register: {rollback_e}")

        except mysql.connector.Error as e:
            logger.error(f"Registration database failed for {email}: {e}")
            flash('An error occurred during registration. Please try again.', 'error')  # Generic user message
            # --- FIX ---
            # Rollback transaction on error
            if conn and conn.is_connected():
                try:
                    conn.rollback()
                    logger.debug("Rolled back transaction after database error during register.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback after database error during register: {rollback_e}")

        except Exception as e:
            logger.error(f"Unexpected error during registration for {email}: {e}", exc_info=True)
            flash('An unexpected error occurred during registration. Please try again.', 'error')
            # --- FIX ---
            # Rollback transaction on unexpected error
            if conn and conn.is_connected():
                try:
                    conn.rollback()
                    logger.debug("Rolled back transaction after unexpected error during register.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback after unexpected error during register: {rollback_e}")

        finally:
            # This finally block is a safeguard if an exception occurred *before*
            # the explicit conn.close() after commit, or if an error occurred
            # between getting the connection and starting the transaction.
            if conn and conn.is_connected():
                logger.warning(f"Connection was still open in register final finally block. Closing now.")
                try:
                    conn.rollback()  # Rollback as a safeguard
                    logger.debug("Rolled back pending transaction in register final finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in register final finally block: {rollback_e}")
                conn.close()
                logger.debug("Database connection closed in register final finally block.")

    return render_template('register.html', form=form)

@app.route('/logout')
def logout():
    # Log logout action
    if 'user_id' in session:
        logger.info(f"User logged out: {session['user_id']} ({session.get('user_name')})")
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    # The require_role decorator handles the initial login check and role verification
    # If the decorator passes, session['user_id'], session['user_name'], and session['user_role'] are guaranteed to exist
    # However, we still need to fetch user details from the DB to ensure the user exists and get the latest role/name.

    if 'user_id' not in session:  # Redundant check if @require_role is used, but harmless
        flash('Please log in.', 'warning')
        return redirect(url_for('login'))

    user_id = session['user_id']
    user_role = session.get('user_role')  # Get role from session initially

    conn1 = None  # Initialize conn1 to None
    conn2 = None  # Initialize conn2 to None
    recent_data = []  # To hold data specific to the role
    template_name = 'dashboard.html'  # Default template

    conn2_user_fetch = None  # Initialize this connection specifically for the initial user fetch

    try:
        # Always fetch user from DB first to verify and get current details
        conn2_user_fetch = get_tenant2()
        c2_user_fetch = conn2_user_fetch.cursor(dictionary=True)
        c2_user_fetch.execute("SELECT name, role FROM users WHERE userID = %s LIMIT 1", (user_id,))
        user = c2_user_fetch.fetchone()

        # --- FIX ---
        # Close connection immediately after fetching user data
        if conn2_user_fetch and conn2_user_fetch.is_connected():
            try:
                conn2_user_fetch.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction after fetching user in dashboard.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback after fetching user in dashboard: {rollback_e}")
            conn2_user_fetch.close()
            logger.debug("Connection conn2_user_fetch closed after fetching user in dashboard.")
            conn2_user_fetch = None  # Set to None

        if not user:
            flash('User account not found. Please register or log in again.', 'error')
            session.clear()  # Clear session if user is not found
            return redirect(url_for('login'))

        # Update session info with fresh data from DB
        session['user_name'] = user['name']
        session['user_role'] = user['role']
        user_role = user['role']  # Use the role from the DB for logic below

        # Fetch data based on the user's role
        if user_role == 'customer':
            conn1 = get_tenant1()  # Acquire connection for tenant1
            c1 = conn1.cursor(dictionary=True)
            c1.execute("SELECT * FROM orders WHERE customerID = %s ORDER BY date DESC LIMIT 5", (user_id,))
            recent_data = c1.fetchall()  # recent_orders
            # template_name remains 'dashboard.html' or specify 'customer_dashboard.html'
        elif user_role == 'sales':
            conn2 = get_tenant2()  # Acquire connection for tenant2
            c2 = conn2.cursor(dictionary=True)
            c2.execute("SELECT * FROM inquiries ORDER BY submitted_at DESC LIMIT 5")
            recent_data = c2.fetchall()  # recent_inquiries
            # template_name remains 'dashboard.html' or specify 'sales_dashboard.html'
        elif user_role in ['finance', 'investor']:
            conn2 = get_tenant2()  # Acquire connection for tenant2
            c2 = conn2.cursor(dictionary=True)
            c2.execute("SELECT * FROM income_statements ORDER BY month DESC LIMIT 5")
            recent_data = c2.fetchall()  # recent_income_statements
            # template_name remains 'dashboard.html' or specify 'finance_investor_dashboard.html'
        elif user_role == 'developer':
            conn2 = get_tenant2()  # Acquire connection for tenant2
            c2 = conn2.cursor(dictionary=True)
            c2.execute("SELECT * FROM files ORDER BY uploaded_at DESC LIMIT 5")
            recent_data = c2.fetchall()  # recent_files
            # template_name remains 'dashboard.html' or specify 'developer_dashboard.html'
        else:
            # Handle unexpected roles (should be caught by require_role, but as a safeguard)
            flash('Invalid user role.', 'error')
            session.clear()
            return redirect(url_for('login'))

        return render_template(template_name, user={'name': user['name'], 'role': user_role}, recent_data=recent_data, role=user_role)

    except mysql.connector.Error as e:
        logger.error(f"Dashboard database error for user {user_id}: {e}")
        flash("An error occurred while loading your dashboard.", 'error')
        # Return a basic template or redirect on error
        # Pass user info from session as fallback
        return render_template('dashboard.html', user={'name': session.get('user_name'), 'role': session.get('user_role')}, role=session.get('user_role'), recent_data=[])
    except Exception as e:
        logger.error(f"Unexpected error in dashboard for user {user_id}: {e}", exc_info=True)
        flash("An unexpected error occurred while loading your dashboard.", 'error')
        return render_template('dashboard.html', user={'name': session.get('user_name'), 'role': session.get('user_role')}, role=session.get('user_role'), recent_data=[])
    finally:
        # --- FIX ---
        # Ensure connections are closed if they were successfully acquired
        if conn1 and conn1.is_connected():
            try:
                conn1.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during dashboard conn1 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in dashboard conn1 finally block: {rollback_e}")
            conn1.close()
            logger.debug("Connection conn1 closed in dashboard finally block.")

        if conn2 and conn2.is_connected():
            # Note: conn2_user_fetch is closed separately above.
            # This conn2 is only used if the role required fetching from tenant2 again.
            try:
                conn2.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during dashboard conn2 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in dashboard conn2 finally block: {rollback_e}")
            conn2.close()
            logger.debug("Connection conn2 closed in dashboard finally block.")

        if conn2_user_fetch and conn2_user_fetch.is_connected():
            logger.warning(f"Connection conn2_user_fetch was still open in dashboard final finally block. Closing now.")
            try:
                conn2_user_fetch.rollback()  # Rollback as a safeguard
                logger.debug("Rolled back pending transaction in dashboard conn2_user_fetch final finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in dashboard conn2_user_fetch final finally block: {rollback_e}")
            conn2_user_fetch.close()
            logger.debug("Database connection conn2_user_fetch closed in dashboard final finally block.")

@app.route('/products/<productID>')
def product(productID):
    conn = None  # Initialize conn to None
    product = None
    form = AddToCartForm()  # Instantiate the form here
    try:
        conn = get_tenant1()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM products WHERE productID = %s LIMIT 1", (productID,))  # Use LIMIT 1
        product = c.fetchone()

        # --- FIX ---
        # Close connection immediately after fetching product data
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during product fetch finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in product fetch finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after product fetch.")
            conn = None  # Set to None so the final finally doesn't try to close it again

        if not product:
            flash('Product not found.', 'error')
            return redirect(url_for('index'))

    except mysql.connector.Error as e:
        logger.error(f"Product route database error for product {productID}: {e}")
        flash("An error occurred while loading the product.", 'error')
        # --- FIX ---
        # Ensure connection is closed on error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during product error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in product error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after product error.")
            conn = None  # Set to None
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Unexpected error in product route for product {productID}: {e}", exc_info=True)
        flash("An unexpected error occurred while loading the product.", 'error')
        # --- FIX ---
        # Ensure connection is closed on unexpected error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during product unexpected error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in product unexpected error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after product unexpected error.")
            conn = None  # Set to None
        return redirect(url_for('index'))
    finally:
        # This finally block is a safeguard if an exception occurred *before*
        # the explicit conn.close() after fetching product data.
        if conn and conn.is_connected():
            logger.warning(f"Connection was still open in product final finally block. Closing now.")
            try:
                conn.rollback()  # Rollback as a safeguard
                logger.debug("Rolled back pending transaction in product final finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in product final finally block: {rollback_e}")
            conn.close()
            logger.debug("Database connection closed in product final finally block.")

    # Pass the product and the form to the template
    return render_template('product.html', product=product, form=form)

# Changed route to POST only as it's a form submission
@app.route('/add_to_cart/<productID>', methods=['POST'])
@require_role('customer')
def add_to_cart(productID):
    form = AddToCartForm()  # Instantiate the form
    conn = None  # Initialize conn outside the try block
    if form.validate_on_submit():  # This validates the form and CSRF token
        quantity = form.quantity.data
        try:
            conn = get_tenant1()
            c = conn.cursor(dictionary=True)

            # Verify product exists and get stock and price
            c.execute("SELECT productID, name, price, stock FROM products WHERE productID = %s LIMIT 1", (productID,))
            product = c.fetchone()

            if not product:
                flash('Product not found.', 'error')
                logger.warning(f"User {session['user_id']} attempted to add non-existent product {productID} to cart.")
                # --- FIX ---
                # Close connection before redirect
                if conn and conn.is_connected():
                    try:
                        conn.rollback()  # Rollback after read operation
                        logger.debug("Rolled back pending transaction during add_to_cart product not found finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback in add_to_cart product not found finally block: {rollback_e}")
                    conn.close()
                    logger.debug("Connection closed after add_to_cart product not found.")
                    conn = None  # Set to None so the final finally doesn't try to close it again
                return redirect(url_for('index'))  # Redirect to index if product is gone

            if product['stock'] < quantity:
                flash(f"Insufficient stock for {product['name']}. Only {product['stock']} available.", 'warning')
                logger.warning(f"User {session['user_id']} attempted to add {quantity} of product {productID} but only {product['stock']} available.")
                # Redirect back to the product page to show the message
                # --- FIX ---
                # Close connection before redirect
                if conn and conn.is_connected():
                    try:
                        conn.rollback()  # Rollback after read operation
                        logger.debug("Rolled back pending transaction during add_to_cart insufficient stock finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback in add_to_cart insufficient stock finally block: {rollback_e}")
                    conn.close()
                    logger.debug("Connection closed after add_to_cart insufficient stock.")
                    conn = None  # Set to None so the final finally doesn't try to close it again
                return redirect(url_for('product', productID=productID))

            # Add to cart or update quantity
            # We don't start an explicit transaction here as this is a single atomic operation (ON DUPLICATE KEY UPDATE)
            c.execute("INSERT INTO cart (customerID, productID, quantity) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE quantity = quantity + %s",
                      (session['user_id'], productID, quantity, quantity))  # Add quantity instead of replacing
            conn.commit()
            flash(f'{quantity} x {product["name"]} added to cart.', 'success')
            logger.info(f"User {session['user_id']} added {quantity} of product {productID} to cart.")

            # --- FIX ---
            # Close connection after successful commit before redirect
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Commit was successful, rollback won't do anything but is safe
                    logger.debug("Rolled back pending transaction during add_to_cart successful commit finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in add_to_cart successful commit finally block: {rollback_e}")
                conn.close()
                logger.debug("Connection closed after add_to_cart successful commit.")
                conn = None  # Set to None so the final finally doesn't try to close it again

            return redirect(url_for('cart'))

        except mysql.connector.Error as e:
            logger.error(f'Add to cart database error for user {session["user_id"]} and product {productID}: {e}')
            flash('An error occurred while adding to the cart. Please try again.', 'error')
            # Redirect back to the product page on database error
            # --- FIX ---
            # Close connection before redirect after error
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback on error
                    logger.debug("Rolled back transaction during add_to_cart database error finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in add_to_cart database error finally block: {rollback_e}")
                conn.close()
                logger.debug("Connection closed after add_to_cart database error.")
                conn = None  # Set to None

            return redirect(url_for('product', productID=productID))
        except Exception as e:
            logger.error(f'Unexpected error in add_to_cart route for user {session["user_id"]} and product {productID}: {e}', exc_info=True)
            flash('An unexpected error occurred. Please try again.', 'error')
            # --- FIX ---
            # Close connection before redirect after unexpected error
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback on error
                    logger.debug("Rolled back transaction during add_to_cart unexpected error finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in add_to_cart unexpected error finally block: {rollback_e}")
                conn.close()
                logger.debug("Connection closed after add_to_cart unexpected error.")
                conn = None  # Set to None so the final finally doesn't try to close it again
            return redirect(url_for('product', productID=productID))

        finally:
            # This finally block is now primarily a safety net if a return happened
            # before explicit closure in try/except, or for truly unhandled exceptions.
            # In the corrected code, the connection should be closed before each return.
            if conn and conn.is_connected():
                logger.warning(f"Connection was still open in add_to_cart final finally block. Closing now.")
                # Rollback any pending transaction before closing/returning to pool.
                # This is crucial if an error happened between getting the connection and starting the transaction.
                try:
                    conn.rollback()
                    logger.debug("Rolled back pending transaction in add_to_cart final finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in add_to_cart final finally block: {rollback_e}")
                conn.close()
                logger.debug("Database connection closed in add_to_cart final finally block.")

    else:
        # If form validation fails (e.g., CSRF, quantity < 1)
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error with {field}: {error}", 'error')
        # Redirect back to the product page
        return redirect(url_for('product', productID=productID))

@app.route('/cart')
@require_role('customer')
def cart():
    conn = None  # Initialize conn to None
    cart_items = []
    try:
        conn = get_tenant1()
        c = conn.cursor(dictionary=True)
        # Join cart with products to get details and calculate subtotal
        c.execute("""
            SELECT c.productID, c.quantity, p.name, p.price, p.image_path
            FROM cart c
            JOIN products p ON c.productID = p.productID
            WHERE c.customerID = %s
        """, (session['user_id'],))
        cart_items = c.fetchall()
        # logger.debug(f"Fetched {len(cart_items)} cart items for user {session['user_id']}")

    except mysql.connector.Error as e:
        logger.error(f'Cart database error for user {session["user_id"]}: {e}')
        flash('An error occurred while loading your cart.', 'error')
    except Exception as e:
        logger.error(f'Unexpected error loading cart for user {session["user_id"]}: {e}', exc_info=True)
        flash('An unexpected error occurred while loading your cart.', 'error')
    finally:
        # --- FIX ---
        # Ensure connection is closed if successfully acquired
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during cart finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in cart finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed in cart finally block.")

    # Calculate cart total in Python
    cart_total = sum(item['quantity'] * item['price'] for item in cart_items)

    return render_template('cart.html', cart_items=cart_items, cart_total=cart_total)

# Add a route to remove items from the cart (optional but good practice)
@app.route('/remove_from_cart/<productID>', methods=['POST'])
@require_role('customer')
def remove_from_cart(productID):
    # You might want a simple form or just handle it via POST with CSRF
    # For simplicity here, assuming a POST request with CSRF token
    conn = None  # Initialize conn to None
    try:
        conn = get_tenant1()
        c = conn.cursor()
        # Check if the item is in the cart for this user before deleting (optional but safer)
        # No explicit transaction needed for a single DELETE statement if autocommit is on
        c.execute("DELETE FROM cart WHERE customerID = %s AND productID = %s", (session['user_id'], productID))
        conn.commit()
        if c.rowcount > 0:
            flash('Item removed from cart.', 'success')
            logger.info(f"User {session['user_id']} removed product {productID} from cart.")
        else:
            flash('Item not found in your cart.', 'warning')
            logger.warning(f"Attempted to remove non-existent item {productID} from cart for user {session['user_id']}")

    except mysql.connector.Error as e:
        logger.error(f'Remove from cart database error for user {session["user_id"]} and product {productID}: {e}')
        flash('An error occurred while removing the item.', 'error')
        # --- FIX ---
        # Rollback transaction on error
        if conn and conn.is_connected():
            try:
                conn.rollback()
                logger.debug("Rolled back transaction after database error during remove_from_cart.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback after database error during remove_from_cart: {rollback_e}")
    except Exception as e:
        logger.error(f'Unexpected error removing from cart for user {session["user_id"]} and product {productID}: {e}', exc_info=True)
        flash('An unexpected error occurred while removing the item.', 'error')
        # --- FIX ---
        # Rollback transaction on unexpected error
        if conn and conn.is_connected():
            try:
                conn.rollback()
                logger.debug("Rolled back transaction after unexpected error during remove_from_cart.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback after unexpected error during remove_from_cart: {rollback_e}")

    finally:
        # --- FIX ---
        # Ensure connection is closed if successfully acquired
        if conn and conn.is_connected():
            # If commit failed, rollback. If succeeded, rollback does nothing.
            try:
                conn.rollback()
                logger.debug("Rolled back pending transaction during remove_from_cart finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in remove_from_cart finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed in remove_from_cart finally block.")

    return redirect(url_for('cart'))

@app.route('/checkout', methods=['GET', 'POST'])
@require_role('customer')
def checkout():
    form = CheckoutForm()  # Instantiate the checkout form
    conn = None  # Initialize conn to None
    cart_items = []
    cart_total = 0

    try:
        conn = get_tenant1()
        c = conn.cursor(dictionary=True)

        # Fetch cart items to display on the checkout page (for GET request)
        # And to process for the POST request
        c.execute("""
            SELECT c.productID, c.quantity, p.name, p.price, p.stock
            FROM cart c
            JOIN products p ON c.productID = p.productID
            WHERE c.customerID = %s
        """, (session['user_id'],))
        cart_items = c.fetchall()

        if not cart_items:
            flash('Your cart is empty.', 'warning')
            logger.warning(f"Checkout attempted with empty cart for user {session['user_id']}.")
            # --- FIX ---
            # Close connection before redirect
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback after read operation
                    logger.debug("Rolled back pending transaction when cart is empty in checkout finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback when cart is empty in checkout finally block: {rollback_e}")
                conn.close()
                logger.debug("Connection closed when cart is empty in checkout.")
                conn = None  # Set to None
            return redirect(url_for('cart'))

        cart_total = sum(item['quantity'] * item['price'] for item in cart_items)

        # --- FIX ---
        # Close connection after fetching cart items if this is a GET request
        # If it's a POST, we need to keep the connection open for the transaction
        if request.method == 'GET':
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Rollback after read operation
                    logger.debug("Rolled back pending transaction after fetching cart items for GET checkout.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback after fetching cart items for GET checkout: {rollback_e}")
                conn.close()
                logger.debug("Connection closed after fetching cart items for GET checkout.")
                conn = None  # Set to None
            return render_template('checkout.html', form=form, cart_items=cart_items, cart_total=cart_total)

        # Process the form submission if it's a POST request and validation passes
        if form.validate_on_submit() and request.method == 'POST':
            # Your existing checkout logic goes here
            # Check stock again before processing the order

            # --- FIX ---
            # If this is a POST request, and conn was closed after the initial fetch (which it shouldn't be now),
            # re-acquire the connection. If the GET block wasn't executed (because it's a POST), conn should
            # still hold the connection from the initial fetch.
            #---if conn is None:  # Should not happen with the logic above, but as a safeguard
                logger.warning("Re-acquiring connection in checkout POST block as it was unexpectedly None.")
                conn = get_tenant1()
                c = conn.cursor(dictionary=True)  # Re-create cursor if conn was re-acquired
                # Note: cart_items is already available from the initial fetch, no need to re-fetch here.

            #---conn.start_transaction()  # Start a transaction for atomic operations
            #---logger.debug("Transaction started for checkout.")
            #---try:
                orderID = str(uuid.uuid4())
                date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                order_total = 0  # Recalculate total based on current prices/quantities

                # Use the same cursor 'c' or create a new one from the existing connection 'conn'
                # Creating a new cursor can sometimes help with clarity, but the original is fine.
                # Let's stick with the original 'c' for simplicity since it's tied to 'conn'.

                for item in cart_items:
                    # Re-fetch product details within the transaction for up-to-date stock/price
                    c.execute("SELECT price, stock, name FROM products WHERE productID = %s FOR UPDATE", (item['productID'],))  # Lock the row
                    product = c.fetchone()

                    if not product or product['stock'] < item['quantity']:
                        # Rollback transaction if stock is insufficient for ANY item
                        conn.rollback()
                        logger.debug("Transaction rolled back due to insufficient stock.")
                        flash(f"Not enough stock for {product.get('name', item['productID'])}. Available: {product.get('stock', 0)}.", 'danger')
                        logger.warning(f"Checkout failed for user {session['user_id']} due to insufficient stock for product {item['productID']}.")
                        # --- FIX ---
                        # Close connection after rollback before returning
                        if conn and conn.is_connected():
                            try:
                                conn.rollback()  # Ensure rollback before closing (redundant but safe)
                                logger.debug("Rolled back pending transaction after stock error in checkout.")
                            except mysql.connector.Error as rollback_e:
                                logger.warning(f"Error during rollback after stock error in checkout: {rollback_e}")
                            conn.close()
                            logger.debug("Connection closed after stock error in checkout.")
                            conn = None  # Set to None so the final finally doesn't try to close again
                        # Need to re-fetch cart_items if rendering the template again?
                        # No, cart_items is already available from the initial fetch.
                        return render_template('checkout.html', form=form, cart_items=cart_items, cart_total=cart_total)

                    # Add item to order_items
                    c.execute("INSERT INTO order_items (orderID, productID, quantity, unit_price) VALUES (%s, %s, %s, %s)",
                              (orderID, item['productID'], item['quantity'], product['price']))
                    order_total += product['price'] * item['quantity']

                    # Update product stock
                    c.execute("UPDATE products SET stock = stock - %s WHERE productID = %s", (item['quantity'], item['productID']))

                # Insert the order
                c.execute("INSERT INTO orders (orderID, customerID, date, total, status) VALUES (%s, %s, %s, %s, %s)",
                           (orderID, session['user_id'], date, order_total, 'Pending'))

                # Clear the cart for the customer
                c.execute("DELETE FROM cart WHERE customerID = %s", (session['user_id'],))

                conn.commit()  # Commit the transaction
                logger.debug("Transaction committed for checkout.")
                flash('Order placed successfully!', 'success')
                logger.info(f"Order {orderID} placed successfully by user {session['user_id']}.")
                # --- FIX ---
                # Close connection after successful commit before redirect
                if conn and conn.is_connected():
                    try:
                        conn.rollback()  # Commit was successful, rollback won't do anything but is safe
                        logger.debug("Rolled back pending transaction after successful checkout commit.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after successful checkout commit: {rollback_e}")
                    conn.close()
                    logger.debug("Connection closed after successful checkout commit.")
                    conn = None  # Set to None so the final finally doesn't try to close again
                return redirect(url_for('order_history'))

            #except mysql.connector.Error as e:
                # --- FIX ---
                # Ensure rollback happens here if an error occurred after transaction started
                if conn and conn.is_connected():
                    try:
                        conn.rollback()
                        logger.debug("Transaction rolled back due to database error during checkout processing.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after database error during checkout processing: {rollback_e}")

                logger.error(f'Checkout processing database error for user {session["user_id"]}: {e}')
                flash('An error occurred while processing your order. Please try again.', 'error')
                # --- FIX ---
                # Close connection before returning on error
                if conn and conn.is_connected():  # Check if conn is still valid after rollback attempt
                    try:
                        conn.rollback()  # Ensure rollback before closing (redundant but safe)
                        logger.debug("Rolled back pending transaction during checkout processing database error finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback during checkout processing database error finally block: {rollback_e}")
                    conn.close()
                    logger.debug("Connection closed after checkout processing database error.")
                    conn = None  # Set to None so the final finally doesn't try to close again

                # Remain on checkout page or redirect to cart
                return render_template('checkout.html', form=form, cart_items=cart_items, cart_total=cart_total)

            #except Exception as e:
                logger.error(f'Unexpected error during checkout processing for user {session["user_id"]}: {e}', exc_info=True)
                flash('An unexpected error occurred while processing your order. Please try again.', 'error')
                # --- FIX ---
                # Ensure rollback happens here if an error occurred after transaction started
                if conn and conn.is_connected():
                    try:
                        conn.rollback()
                        logger.debug("Transaction rolled back due to unexpected error during checkout processing.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after unexpected error during checkout processing: {rollback_e}")
                # --- FIX ---
                # Close connection before returning on error
                if conn and conn.is_connected():  # Check if conn is still valid after rollback attempt
                    try:
                        conn.rollback()  # Ensure rollback before closing (redundant but safe)
                        logger.debug("Rolled back pending transaction during checkout processing unexpected error finally block.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback during checkout processing unexpected error finally block: {rollback_e}")
                    conn.close()
                    logger.debug("Connection closed after checkout processing unexpected error.")
                    conn = None  # Set to None so the final finally doesn't try to close again

                return render_template('checkout.html', form=form, cart_items=cart_items, cart_total=cart_total)

    except mysql.connector.Error as e:
        # This catch block primarily handles errors during the initial cart item fetch (for GET or POST)
        logger.error(f'Checkout load database error for user {session["user_id"]}: {e}')
        flash('An error occurred while loading checkout details.', 'error')
        # Redirect to cart or index if loading fails
        # --- FIX ---
        # Ensure connection is closed on this error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back transaction after checkout load database error.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback after checkout load database error: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after checkout load database error.")
            conn = None  # Set to None
        return redirect(url_for('cart'))
    except Exception as e:
        logger.error(f'Unexpected error loading checkout for user {session["user_id"]}: {e}', exc_info=True)
        flash('An unexpected error occurred while loading checkout details.', 'error')
        # --- FIX ---
        # Ensure connection is closed on this error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back transaction after unexpected error loading checkout.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback after unexpected error loading checkout: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after unexpected error loading checkout.")
            conn = None  # Set to None
        return redirect(url_for('cart'))

    finally:
        # This finally block is a safety net.
        # If the POST request processing reached its own finally block or return,
        # the connection would have already been closed there (and conn set to None).
        # This block handles cases where the initial GET request finished, or
        # if an error occurred *before* the form.validate_on_submit() check in POST,
        # or any unhandled exception that wasn't caught specifically above.
        if conn and conn.is_connected():
            logger.warning(f"Connection was still open in checkout final finally block. Closing now.")
            # Rollback any pending transaction before closing/returning to pool.
            # This is crucial if an error happened between getting the connection and starting the transaction.
            try:
                conn.rollback()
                logger.debug("Rolled back any pending transaction before closing connection in checkout final finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in checkout final finally block: {rollback_e}")
            conn.close()
            logger.debug("Database connection closed in checkout final finally block.")

    # For GET request or form validation failure (e.g., invalid card details)
    # cart_items and cart_total will already be fetched
    if not form.validate_on_submit() and request.method == 'POST':
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error with {field}: {error}", 'error')
    return render_template('checkout.html', form=form, cart_items=cart_items, cart_total=cart_total)

@app.route('/order_history')
@require_role('customer')
def order_history():
    conn = None  # Initialize conn to None
    orders = []
    try:
        conn = get_tenant1()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM orders WHERE customerID = %s ORDER BY date DESC", (session['user_id'],))
        orders = c.fetchall()
        # logger.debug(f"Fetched {len(orders)} orders for user {session['user_id']}")
    except mysql.connector.Error as e:
        logger.error(f'Order history database error for user {session["user_id"]}: {e}')
        flash('An error occurred while loading your order history.', 'error')
    except Exception as e:
        logger.error(f'Unexpected error loading order history for user {session["user_id"]}: {e}', exc_info=True)
        flash('An unexpected error occurred while loading your order history.', 'error')
    finally:
        # --- FIX ---
        # Ensure connection is closed if successfully acquired
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during order_history finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in order_history finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed in order_history finally block.")

    return render_template('order_history.html', orders=orders)

@app.route('/update_order_status', methods=['POST'])
@require_role('sales')
def update_order_status():
    orderID = request.form.get('orderID')
    status = request.form.get('status')
    conn = None
    try:
        conn = get_tenant1()
        c = conn.cursor(dictionary=True)
        c.execute("UPDATE orders SET status = %s WHERE orderID = %s", (status, orderID))
        conn.commit()
        flash('Order status updated successfully.', 'success')
    except mysql.connector.Error as e:
        logger.error(f'Order status update error: {e}')
        flash(f'Order status update error: {e}', 'error')
    finally:
        if conn and conn.is_connected():
            conn.close()
    return redirect(url_for('sales'))


@app.route('/inquiries', methods=['GET', 'POST'])
def inquiries():
    form = InquiryForm()
    conn = None  # Initialize conn outside try block
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        message = form.message.data
        inquiryID = str(uuid.uuid4())
        submitted_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        auto_response = auto_respond_query(message)
        status = 'complete' if auto_response else 'pending'

        try:
            conn = get_tenant2()
            c = conn.cursor()
            # No explicit transaction needed for a single INSERT statement if autocommit is on
            c.execute("INSERT INTO inquiries (inquiryID, name, email, message, submitted_at, status, response) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                      (inquiryID, name, email, message, submitted_at, status, auto_response))
            conn.commit()
            logger.info(f"Inquiry submitted by {email}, ID: {inquiryID}, Status: {status}")
            flash('Inquiry submitted successfully!' + (' Auto-response sent.' if auto_response else ''), 'success')
            # --- FIX ---
            # Close connection after successful commit before redirect
            if conn and conn.is_connected():
                try:
                    conn.rollback()  # Commit was successful, rollback won't do anything but is safe
                    logger.debug("Rolled back pending transaction during inquiries successful commit finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in inquiries successful commit finally block: {rollback_e}")
                conn.close()
                logger.debug("Connection closed after inquiries successful commit.")
                conn = None  # Set to None

            return redirect(url_for('inquiries'))  # Redirect after POST
        except mysql.connector.Error as e:
            logger.error(f'Inquiry submission database error: {e}')
            flash('An error occurred while submitting your inquiry. Please try again.', 'error')
            # --- FIX ---
            # Rollback transaction on error
            if conn and conn.is_connected():
                try:
                    conn.rollback()
                    logger.debug("Rolled back transaction after database error during inquiries submission.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback after database error during inquiries submission: {rollback_e}")

        except Exception as e:
            logger.error(f'Unexpected error during inquiry submission: {e}', exc_info=True)
            flash('An unexpected error occurred while submitting your inquiry. Please try again.', 'error')
            # --- FIX ---
            # Rollback transaction on unexpected error
            if conn and conn.is_connected():
                try:
                    conn.rollback()
                    logger.debug("Rolled back transaction after unexpected error during inquiries submission.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback after unexpected error during inquiries submission: {rollback_e}")

        finally:
            # --- FIX ---
            # Ensure connection is closed if successfully acquired
            if conn and conn.is_connected():
                # If commit failed, rollback. If succeeded, rollback does nothing.
                try:
                    conn.rollback()
                    logger.debug("Rolled back pending transaction during inquiries finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in inquiries finally block: {rollback_e}")
                conn.close()
                logger.debug("Connection closed in inquiries finally block.")

    # For GET request or failed POST validation
    return render_template('inquiries.html', form=form)

@app.route('/sales', methods=['GET', 'POST'])
@require_role('sales')
def sales():
    response_form = SalesResponseForm()  # Instantiate the response form
    conn1 = None  # Initialize conn1 to None
    conn2 = None  # Initialize conn2 to None
    inquiries = []
    orders = []

    try:
        # Check if this is a POST request for the response form
        if response_form.validate_on_submit() and request.method == 'POST':
            inquiryID = response_form.inquiryID.data
            response_text = response_form.response.data
            conn2 = get_tenant2()  # Get connection for update
            c2 = conn2.cursor(dictionary=True)

            # Verify the inquiry exists and is pending before updating (optional but safer)
            # No explicit transaction needed for a single UPDATE statement if autocommit is on
            c2.execute("SELECT inquiryID, status FROM inquiries WHERE inquiryID = %s LIMIT 1", (inquiryID,))
            inquiry_to_update = c2.fetchone()

            if inquiry_to_update and inquiry_to_update['status'] == 'pending':
                c2.execute("UPDATE inquiries SET response = %s, status = 'complete' WHERE inquiryID = %s", (response_text, inquiryID))
                conn2.commit()
                flash('Response submitted.', 'success')
                logger.info(f"Sales user {session['user_id']} responded to inquiry {inquiryID}")
            elif inquiry_to_update:
                flash('This inquiry has already been responded to.', 'warning')
                logger.warning(f"Sales user {session['user_id']} attempted to respond to completed inquiry {inquiryID}")
                # --- FIX ---
                # Rollback transaction on warning/info condition (no DB change, but safe)
                if conn2 and conn2.is_connected():
                    try:
                        conn2.rollback()
                        logger.debug("Rolled back transaction after sales response warning.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after sales response warning: {rollback_e}")
            else:
                flash('Inquiry not found.', 'error')
                logger.warning(f"Sales user {session['user_id']} attempted to respond to non-existent inquiry {inquiryID}")
                # --- FIX ---
                # Rollback transaction on error condition (no DB change, but safe)
                if conn2 and conn2.is_connected():
                    try:
                        conn2.rollback()
                        logger.debug("Rolled back transaction after sales response error.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after sales response error: {rollback_e}")

            # --- FIX ---
            # Close connection after handling POST before redirect
            if conn2 and conn2.is_connected():
                try:
                    conn2.rollback()  # If commit failed, rollback. If succeeded, rollback does nothing.
                    logger.debug("Rolled back pending transaction after sales response POST.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback after sales response POST: {rollback_e}")
                conn2.close()
                logger.debug("Connection closed after sales response POST.")
                conn2 = None  # Set to None so finally doesn't try to close again

            return redirect(url_for('sales'))  # Redirect after POST

        # For GET request or failed POST validation
        # Fetch data for display
        # Acquire connections for fetching data
        conn2 = get_tenant2()  # Get connection for fetching inquiries
        c2 = conn2.cursor(dictionary=True)
        c2.execute("SELECT * FROM inquiries ORDER BY submitted_at DESC")
        inquiries = c2.fetchall()

        # --- FIX ---
        # Close conn2 after fetching inquiries if no more tenant2 operations are needed
        if conn2 and conn2.is_connected():
            try:
                conn2.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction after fetching inquiries in sales.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback after fetching inquiries in sales: {rollback_e}")
            conn2.close()
            logger.debug("Connection conn2 closed after fetching inquiries in sales.")
            conn2 = None  # Set to None so finally doesn't try to close again

        conn1 = get_tenant1()  # Get connection for fetching orders
        c1 = conn1.cursor(dictionary=True)
        c1.execute("SELECT * FROM orders ORDER BY date DESC")
        orders = c1.fetchall()

        # logger.debug(f"Fetched {len(inquiries)} inquiries and {len(orders)} orders for sales dashboard.")

    except mysql.connector.Error as e:
        logger.error(f'Sales database error for user {session["user_id"]}: {e}')
        flash('An error occurred while loading sales data.', 'error')
    except Exception as e:
        logger.error(f'Unexpected error loading sales data for user {session["user_id"]}: {e}', exc_info=True)
        flash('An unexpected error occurred while loading sales data.', 'error')
    finally:
        # --- FIX ---
        # Ensure connections are closed if they were successfully acquired
        if conn1 and conn1.is_connected():
            try:
                conn1.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during sales conn1 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in sales conn1 finally block: {rollback_e}")
            conn1.close()
            logger.debug("Connection conn1 closed in sales finally block.")

        if conn2 and conn2.is_connected():
            # Note: conn2 used for POST is closed separately above.
            # This conn2 is only used if the GET block was executed.
            try:
                conn2.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during sales conn2 finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in sales conn2 finally block: {rollback_e}")
            conn2.close()
            logger.debug("Connection conn2 closed in sales finally block.")

    # Pass the response form and data to the template
    return render_template('sales.html', response_form=response_form, inquiries=inquiries, orders=orders)

@app.route('/finance', methods=['GET', 'POST'])
@require_role('finance')
def finance():
    form = IncomeStatementForm()
    conn = None  # Initialize conn to None
    statements = []

    # Check if this is a POST request for the income statement form
    if form.validate_on_submit() and request.method == 'POST':
        month = form.month.data
        revenue = form.revenue.data
        expenses = form.expenses.data
        net_income = revenue - expenses
        statementID = str(uuid.uuid4())

        conn_check = None  # Initialize conn_check to None
        try:
            # Check if statement for this month already exists (optional)
            conn_check = get_tenant2()
            c_check = conn_check.cursor()
            c_check.execute("SELECT COUNT(*) FROM income_statements WHERE month = %s", (month,))
            count = c_check.fetchone()[0]

            # --- FIX ---
            # Close conn_check after use
            if conn_check and conn_check.is_connected():
                try:
                    conn_check.rollback()  # Rollback after read operation
                    logger.debug("Rolled back pending transaction during finance month check finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback in finance month check finally block: {rollback_e}")
                conn_check.close()
                logger.debug("Connection conn_check closed after finance month check.")
                conn_check = None  # Set to None

            if count > 0:
                flash(f'Income statement for {month} already exists.', 'warning')
                # Re-fetch statements to show existing ones (handled by the GET block below)
                # Skip the rest of the POST logic and let the GET block execute
                pass  # Continue to the fetch block below

            else:  # Month does not exist, proceed with insert
                chart_data = {
                    'statementID': statementID,
                    'month': month,
                    'revenue': revenue,
                    'expenses': expenses,
                    'net_income': net_income
                }
                chart_path = generate_income_chart(chart_data)

                if chart_path:  # Only insert if chart generation was successful
                    try:
                        conn = get_tenant2()  # Get connection for insert
                        c = conn.cursor()
                        # No explicit transaction needed for a single INSERT statement if autocommit is on
                        c.execute("INSERT INTO income_statements (statementID, month, revenue, expenses, net_income, chart_path) VALUES (%s, %s, %s, %s, %s, %s)",
                                  (statementID, month, revenue, expenses, net_income, chart_path))
                        conn.commit()
                        logger.info(f"Income statement created for month: {month}")
                        flash('Income statement created.', 'success')
                    except mysql.connector.Error as e:
                        logger.error(f'Finance database error during insert: {e}')
                        flash('An error occurred while saving the income statement. Please try again.', 'error')
                        # Clean up the generated chart file if DB insertion failed
                        if os.path.exists(chart_path):
                            try:
                                os.remove(chart_path)
                                logger.debug(f"Cleaned up chart file {chart_path} after DB error.")
                            except OSError as cleanup_e:
                                logger.error(f"Error cleaning up chart file {chart_path}: {cleanup_e}")
                        # --- FIX ---
                        # Rollback transaction on error
                        if conn and conn.is_connected():
                            try:
                                conn.rollback()
                                logger.debug("Rolled back transaction after database error during finance insert.")
                            except mysql.connector.Error as rollback_e:
                                logger.warning(f"Error during rollback after database error during finance insert: {rollback_e}")

                    except Exception as e:
                        logger.error(f'Unexpected error during finance insert: {e}', exc_info=True)
                        flash('An unexpected error occurred while saving the income statement. Please try again.', 'error')
                        # Clean up the generated chart file if DB insertion failed
                        if os.path.exists(chart_path):
                            try:
                                os.remove(chart_path)
                                logger.debug(f"Cleaned up chart file {chart_path} after unexpected error.")
                            except OSError as cleanup_e:
                                logger.error(f"Error cleaning up chart file {chart_path}: {cleanup_e}")
                        # --- FIX ---
                        # Rollback transaction on unexpected error
                        if conn and conn.is_connected():
                            try:
                                conn.rollback()
                                logger.debug("Rolled back transaction after unexpected error during finance insert.")
                            except mysql.connector.Error as rollback_e:
                                logger.warning(f"Error during rollback after unexpected error during finance insert: {rollback_e}")

                    finally:
                        # --- FIX ---
                        # Ensure connection is closed if successfully acquired for insert
                        if conn and conn.is_connected():
                            # If commit failed, rollback. If succeeded, rollback does nothing.
                            try:
                                conn.rollback()
                                logger.debug("Rolled back pending transaction after finance insert.")
                            except mysql.connector.Error as rollback_e:
                                logger.warning(f"Error during rollback after finance insert: {rollback_e}")
                            conn.close()
                            logger.debug("Connection closed after finance insert.")
                            conn = None

                else:
                    flash('Failed to generate income statement chart.', 'error')

        except mysql.connector.Error as e:
            logger.error(f"Finance month check database error: {e}")
            flash('An error occurred during month check.', 'error')
            # --- FIX ---
            # Close conn_check on error
            if conn_check and conn_check.is_connected():
                try:
                    conn_check.rollback()  # Rollback on error
                    logger.debug("Rolled back pending transaction on error in finance month check finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback on error in finance month check finally block: {rollback_e}")
                conn_check.close()
                logger.debug("Connection conn_check closed on error after finance month check.")
                conn_check = None  # Set to None
        except Exception as e:
            logger.error(f"Unexpected error during finance month check: {e}", exc_info=True)
            flash('An unexpected error occurred during month check.', 'error')
            # --- FIX ---
            # Close conn_check on unexpected error
            if conn_check and conn_check.is_connected():
                try:
                    conn_check.rollback()  # Rollback on error
                    logger.debug("Rolled back pending transaction on unexpected error in finance month check finally block.")
                except mysql.connector.Error as rollback_e:
                    logger.warning(f"Error during rollback on unexpected error in finance month check finally block: {rollback_e}")
                conn_check.close()
                logger.debug("Connection conn_check closed on unexpected error after finance month check.")
                conn_check = None  # Set to None

    # For GET request or failed POST/DB insert
    # Fetch statements for display
    # This block executes for GET requests or if the POST logic didn't redirect
    conn = None  # Initialize conn to None before fetching
    try:
        conn = get_tenant2()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM income_statements ORDER BY month DESC")
        statements = c.fetchall()
        # logger.debug(f"Fetched {len(statements)} income statements for finance dashboard.")
    except mysql.connector.Error as e:
        logger.error(f'Finance database error during fetch: {e}')
        flash('An error occurred while loading income statements.', 'error')
    except Exception as e:
        logger.error(f'Unexpected error loading income statements for finance: {e}', exc_info=True)
        flash('An unexpected error occurred while loading income statements.', 'error')
    finally:
        # --- FIX ---
        # Ensure connection is closed if successfully acquired for fetch
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during finance fetch finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in finance fetch finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after finance fetch.")

    return render_template('finance.html', form=form, statements=statements)

@app.route('/investor')
@require_role('investor')
def investor():
    conn = None  # Initialize conn to None
    statements = []
    try:
        conn = get_tenant2()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM income_statements ORDER BY month DESC")
        statements = c.fetchall()
        # logger.debug(f"Fetched {len(statements)} income statements for investor dashboard.")
    except mysql.connector.Error as e:
        logger.error(f'Investor database error: {e}')
        flash('An error occurred while loading income statements.', 'error')
    except Exception as e:
        logger.error(f'Unexpected error loading income statements for investor: {e}', exc_info=True)
        flash('An unexpected error occurred while loading income statements.', 'error')
    finally:
        # --- FIX ---
        # Ensure connection is closed if successfully acquired
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during investor finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in investor finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed in investor finally block.")

    return render_template('investor.html', statements=statements)

@app.route('/developer', methods=['GET', 'POST'])
@require_role('developer')
def developer():
    conn = None  # Initialize conn to None
    files = []  # Initialize files list

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename and allowed_file(file.filename):  # Check if file exists and has a filename
            fileID = str(uuid.uuid4())
            # Sanitize filename to prevent directory traversal attacks
            filename_safe = secure_filename(file.filename)  # secure_filename is imported
            # Add fileID to the filename to make it unique on the filesystem
            filename_on_disk = f"{fileID}_{filename_safe}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_on_disk)

            try:
                # Ensure the upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(file_path)  # Save the file first

                conn = get_tenant2()  # Get connection for insert
                c = conn.cursor()
                # No explicit transaction needed for a single INSERT statement if autocommit is on
                # Store the original filename in the DB, but the path to the file on disk
                c.execute("INSERT INTO files (fileID, name, path, uploaded_by, uploaded_at) VALUES (%s, %s, %s, %s, %s)",
                          (fileID, file.filename, file_path, session['user_id'], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                flash('File uploaded successfully.', 'success')
                logger.info(f"User {session['user_id']} uploaded file {file.filename} (ID: {fileID})")
                # --- FIX ---
                # Close connection after successful commit before redirect
                if conn and conn.is_connected():
                    try:
                        conn.rollback()  # Commit was successful, rollback won't do anything but is safe
                        logger.debug("Rolled back pending transaction after developer upload commit.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after developer upload commit: {rollback_e}")
                    conn.close()
                    logger.debug("Connection closed after developer upload commit.")
                    conn = None  # Set to None

                return redirect(url_for('developer'))  # Redirect after successful upload
            except mysql.connector.Error as e:
                logger.error(f'File upload database error: {e}')
                flash('An error occurred while saving file information.', 'error')
                # Clean up the uploaded file if DB insertion failed
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.debug(f"Cleaned up uploaded file {file_path} after DB error.")
                    except OSError as cleanup_e:
                        logger.error(f"Error cleaning up uploaded file {file_path}: {cleanup_e}")
                # --- FIX ---
                # Rollback transaction on error
                if conn and conn.is_connected():
                    try:
                        conn.rollback()
                        logger.debug("Rolled back transaction after database error during developer upload.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after database error during developer upload: {rollback_e}")

            except IOError as e:  # Catch errors during file saving
                logger.error(f'File save error: {e}')
                flash('An error occurred while saving the file.', 'error')
                # Note: If file.save fails, conn might not have been acquired yet,
                # so no DB rollback/close is needed for this specific exception.

            except Exception as e:
                logger.error(f'Unexpected error during file upload: {e}', exc_info=True)
                flash('An unexpected error occurred during file upload.', 'error')
                # Clean up the uploaded file if an unexpected error occurred after saving
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.debug(f"Cleaned up uploaded file {file_path} after unexpected error.")
                    except OSError as cleanup_e:
                        logger.error(f"Error cleaning up uploaded file {file_path}: {cleanup_e}")
                # --- FIX ---
                # Rollback transaction on unexpected error
                if conn and conn.is_connected():
                    try:
                        conn.rollback()
                        logger.debug("Rolled back transaction after unexpected error during developer upload.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after unexpected error during developer upload: {rollback_e}")

            finally:
                # --- FIX ---
                # Ensure connection is closed if successfully acquired
                if conn and conn.is_connected():
                    # If commit failed, rollback. If succeeded, rollback does nothing.
                    try:
                        conn.rollback()
                        logger.debug("Rolled back pending transaction after developer upload.")
                    except mysql.connector.Error as rollback_e:
                        logger.warning(f"Error during rollback after developer upload: {rollback_e}")
                    conn.close()
                    logger.debug("Connection closed after developer upload.")

        else:
            flash('No file selected or invalid file type.', 'error')
            # Remain on the developer page

    # For GET request or failed POST
    # Fetch files for display
    conn = None  # Initialize conn to None before fetching
    try:
        conn = get_tenant2()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM files ORDER BY uploaded_at DESC")
        files = c.fetchall()
        # logger.debug(f"Fetched {len(files)} files for developer dashboard.")
    except mysql.connector.Error as e:
        logger.error(f'Developer database error during fetch: {e}')
        flash('An error occurred while loading files.', 'error')
    except Exception as e:
        logger.error(f'Unexpected error loading files for developer: {e}', exc_info=True)
        flash('An unexpected error occurred while loading files.', 'error')
    finally:
        # --- FIX ---
        # Ensure connection is closed if successfully acquired for fetch
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction during developer fetch finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in developer fetch finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after developer fetch.")

    return render_template('developer.html', files=files)

@app.route('/download_file/<fileID>')
@require_role('developer')
def download_file(fileID):
    conn = None  # Initialize conn to None
    file = None
    try:
        conn = get_tenant2()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT name, path FROM files WHERE fileID = %s LIMIT 1", (fileID,))  # Use LIMIT 1
        file = c.fetchone()

        # --- FIX ---
        # Close connection immediately after fetching file path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction after fetching file path for download.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback after fetching file path for download: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after fetching file path for download.")
            conn = None  # Set to None

        if file and os.path.exists(file['path']):  # Check if file exists on the filesystem
            logger.info(f"User {session['user_id']} downloading file {fileID} ({file['name']})")
            # send_file handles closing the file object
            return send_file(file['path'], download_name=file['name'])
        elif file:
            logger.warning(f"File record found for ID {fileID} but file path {file['path']} does not exist.")
            flash('File not found on server.', 'error')
        else:
            flash('File not found in database.', 'error')
            logger.warning(f"Download attempt for non-existent file ID {fileID} by user {session['user_id']}")

    except mysql.connector.Error as e:
        logger.error(f'File download database error for ID {fileID}: {e}')
        flash('An error occurred while fetching file details.', 'error')
        # --- FIX ---
        # Ensure connection is closed on error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during download_file error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in download_file error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after download_file error.")
            conn = None  # Set to None
    except Exception as e:  # Catch other potential errors during send_file or os.path.exists
        logger.error(f'File download error for ID {fileID}: {e}', exc_info=True)
        flash('An unexpected error occurred during download.', 'error')
        # --- FIX ---
        # Ensure connection is closed on unexpected error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during download_file unexpected error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in download_file unexpected error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after download_file unexpected error.")
            conn = None  # Set to None
    finally:
        # --- FIX ---
        # This final finally block is a safeguard, but the connection should ideally be closed earlier.
        if conn and conn.is_connected():
            logger.warning(f"Connection was still open in download_file final finally block. Closing now.")
            try:
                conn.rollback()  # Rollback as a safeguard
                logger.debug("Rolled back pending transaction in download_file final finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in download_file final finally block: {rollback_e}")
            conn.close()
            logger.debug("Database connection closed in download_file final finally block.")

    return redirect(url_for('developer'))

# API Endpoints (consider adding authentication/authorization for production)
@app.route('/iwc_api/orders')
def iwc_api_orders():
    conn = None  # Initialize conn to None
    try:
        conn = get_tenant1()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM orders")
        orders = c.fetchall()
        # logger.debug(f"API fetched {len(orders)} orders.")

        # --- FIX ---
        # Close connection before returning JSON response
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction before returning iwc_api_orders.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback before returning iwc_api_orders: {rollback_e}")
            conn.close()
            logger.debug("Connection closed before returning iwc_api_orders.")
            conn = None  # Set to None

        return jsonify(orders)
    except mysql.connector.Error as e:
        logger.error(f'IWC API orders database error: {e}')
        # --- FIX ---
        # Ensure connection is closed on error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during iwc_api_orders error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback during iwc_api_orders error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after iwc_api_orders error.")
            conn = None  # Set to None
        return jsonify({'error': 'Database error fetching orders.'}), 500
    except Exception as e:
        logger.error(f'Unexpected error in IWC API orders: {e}', exc_info=True)
        # --- FIX ---
        # Ensure connection is closed on unexpected error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during iwc_api_orders unexpected error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback during iwc_api_orders unexpected error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after iwc_api_orders unexpected error.")
            conn = None  # Set to None
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        # --- FIX ---
        # This final finally block is a safeguard
        if conn and conn.is_connected():
            logger.warning(f"Connection was still open in iwc_api_orders final finally block. Closing now.")
            try:
                conn.rollback()  # Rollback as a safeguard
                logger.debug("Rolled back pending transaction in iwc_api_orders final finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in iwc_api_orders final finally block: {rollback_e}")
            conn.close()
            logger.debug("Database connection closed in iwc_api_orders final finally block.")

@app.route('/iwc_api/income_statements')
def iwc_api_income_statements():
    conn = None  # Initialize conn to None
    try:
        conn = get_tenant2()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM income_statements")
        statements = c.fetchall()
        # logger.debug(f"API fetched {len(statements)} income statements.")

        # --- FIX ---
        # Close connection before returning JSON response
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback after read operation
                logger.debug("Rolled back pending transaction before returning iwc_api_income_statements.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback before returning iwc_api_income_statements: {rollback_e}")
            conn.close()
            logger.debug("Connection closed before returning iwc_api_income_statements.")
            conn = None  # Set to None

        return jsonify(statements)
    except mysql.connector.Error as e:
        logger.error(f'IWC API income statements database error: {e}')
        # --- FIX ---
        # Ensure connection is closed on error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during iwc_api_income_statements error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback during iwc_api_income_statements error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after iwc_api_income_statements error.")
            conn = None  # Set to None
        return jsonify({'error': 'Database error fetching income statements.'}), 500
    except Exception as e:
        logger.error(f'Unexpected error in IWC API income statements: {e}', exc_info=True)
        # --- FIX ---
        # Ensure connection is closed on unexpected error path
        if conn and conn.is_connected():
            try:
                conn.rollback()  # Rollback on error
                logger.debug("Rolled back pending transaction during iwc_api_income_statements unexpected error finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback during iwc_api_income_statements unexpected error finally block: {rollback_e}")
            conn.close()
            logger.debug("Connection closed after iwc_api_income_statements unexpected error.")
            conn = None  # Set to None
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        # --- FIX ---
        # This final finally block is a safeguard
        if conn and conn.is_connected():
            logger.warning(f"Connection was still open in iwc_api_income_statements final finally block. Closing now.")
            try:
                conn.rollback()  # Rollback as a safeguard
                logger.debug("Rolled back pending transaction in iwc_api_income_statements final finally block.")
            except mysql.connector.Error as rollback_e:
                logger.warning(f"Error during rollback in iwc_api_income_statements final finally block: {rollback_e}")
            conn.close()
            logger.debug("Database connection closed in iwc_api_income_statements final finally block.")

if __name__ == '__main__':
    # Initialize connection pools here
    try:
        tenant1_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="tenant1_pool",
            pool_size=5,  # Adjust pool size based on expected load
            pool_reset_session=True,  # Reset session state (like transactions) when returning connection
            connection_timeout=30,
            **TENANT1_CONFIG
        )
        tenant2_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="tenant2_pool",
            pool_size=5,  # Adjust pool size based on expected load
            pool_reset_session=True,  # Reset session state (like transactions) when returning connection
            connection_timeout=30,
            **TENANT2_CONFIG
        )
        logger.info("Connection pools initialized successfully.")
    except mysql.connector.Error as e:
        logger.critical(f"Failed to initialize connection pools: {e}")
        # Exit if database connection cannot be established on startup
        exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error during connection pool initialization: {e}", exc_info=True)
        exit(1)

    # Ensure databases are initialized and tables exist before running the app
    # This block only runs when the script is executed directly
    try:
        init_tenant1()
        init_tenant2()
        verify_tables()
    except Exception as e:  # Catch the specific exceptions raised by init/verify
        logger.critical(f"Application failed to start due to database initialization or verification error: {e}")
        # In a production environment, you might want to halt execution here
        # For this assignment, let's allow the app to run but log the error
        exit(1)  # Exit if DB init/verify fails critically

    # You can set FLASK_ENV=development in your environment
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
