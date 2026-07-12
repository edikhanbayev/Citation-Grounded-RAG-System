# setup.py
from app import app, db
from app.models import User
import sqlalchemy as sa


def boot_and_seed_environment():
    with app.app_context():
        #  REMOVED: db.drop_all() has been removed to preserve data across runs.

        print("Ensuring relational schema frameworks exist...")
        # create_all() safely checks if tables exist first before attempting creation
        db.create_all()

        print("Checking for Master System Administrator status...")
        # Use a safe modern SQLAlchemy scalar query to check if the admin already exists
        admin_exists = db.session.scalar(
            sa.select(User).where(User.username == "system_root")
        )

        if not admin_exists:
            print("Injecting initial Master System Administrator profile context...")
            admin = User(
                username="system_root",
                email="root@university.ac.uk",
                student_id="SYS-ADMIN-01",
                role="admin",
                is_approved=True
            )
            admin.set_password("master123")
            db.session.add(admin)
            db.session.commit()
            print("Master Admin account generated successfully (system_root / master123).")
        else:
            print("Master Admin account already exists. Skipping seed step.")


if __name__ == '__main__':
    # Safely initialize or verify the system tables without wiping them
    boot_and_seed_environment()

    print("Starting local web container server...")
    # Start the web app server live
    app.run(debug=True, port=5000)