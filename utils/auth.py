import hashlib
def check_login(username, password, user_db):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    if username in user_db:
        return user_db[username] == hashed
    return False
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
def register_user(username, password, user_db):
    if username in user_db:
        return False  # User already exists
    user_db[username] = hash_password(password)
    return True
def get_user_db():
    return {
        'admin': hash_password('admin123'),
        'user1': hash_password('password1'),
        'user2': hash_password('password2')
    }

def get_user(username, user_db):
    return user_db.get(username, None)

  