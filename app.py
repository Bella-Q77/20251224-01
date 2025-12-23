from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 用于session加密

# 初始化数据库
def init_db():
    conn = sqlite3.connect('christmas_trees.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trees
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  child_name TEXT UNIQUE,
                  tree_data TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# 登录页面
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'test' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('generate_tree'))
        else:
            return render_template('login.html', error='账号或密码错误')
    return render_template('login.html')

# 圣诞树生成页面
@app.route('/generate', methods=['GET', 'POST'])
def generate_tree():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        child_name = request.form['child_name']
        # 检查是否已存在该姓名的圣诞树
        conn = sqlite3.connect('christmas_trees.db')
        c = conn.cursor()
        c.execute('SELECT tree_data FROM trees WHERE child_name = ?', (child_name,))
        existing_tree = c.fetchone()
        conn.close()
        
        if existing_tree:
            # 存在则返回已保存的树
            return jsonify({'success': True, 'tree_data': existing_tree[0], 'new': False})
        else:
            # 生成新的圣诞树数据
            tree_data = generate_new_tree(child_name)
            # 保存到数据库
            conn = sqlite3.connect('christmas_trees.db')
            c = conn.cursor()
            c.execute('INSERT INTO trees (child_name, tree_data) VALUES (?, ?)', (child_name, tree_data))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'tree_data': tree_data, 'new': True})
    
    return render_template('generate.html')

def generate_new_tree(child_name):
    # 生成圣诞树的HTML结构
    tree_html = f'''<div class="christmas-tree">
        <div class="tree-top"></div>
        <div class="tree-middle"></div>
        <div class="tree-bottom"></div>
        <div class="tree-trunk"></div>
        <div class="child-name">{child_name}</div>
    </div>'''
    return tree_html

# 保存圣诞树
@app.route('/save_tree', methods=['POST'])
def save_tree():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    child_name = request.form['child_name']
    tree_data = request.form['tree_data']
    
    conn = sqlite3.connect('christmas_trees.db')
    c = conn.cursor()
    
    # 检查是否已存在
    c.execute('SELECT id FROM trees WHERE child_name = ?', (child_name,))
    existing = c.fetchone()
    
    if existing:
        # 更新现有记录
        c.execute('UPDATE trees SET tree_data = ? WHERE child_name = ?', (tree_data, child_name))
    else:
        # 插入新记录
        c.execute('INSERT INTO trees (child_name, tree_data) VALUES (?, ?)', (child_name, tree_data))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': '圣诞树已保存成功！'})

# 查看所有圣诞树
@app.route('/view_trees')
def view_trees():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('christmas_trees.db')
    c = conn.cursor()
    c.execute('SELECT * FROM trees ORDER BY created_at DESC')
    trees = c.fetchall()
    conn.close()
    
    return render_template('view_trees.html', trees=trees)

# 获取单个圣诞树
@app.route('/get_tree/<child_name>')
def get_tree(child_name):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('christmas_trees.db')
    c = conn.cursor()
    c.execute('SELECT tree_data FROM trees WHERE child_name = ?', (child_name,))
    tree = c.fetchone()
    conn.close()
    
    if tree:
        return jsonify({'success': True, 'tree_data': tree[0]})
    else:
        return jsonify({'success': False, 'message': '未找到该小朋友的圣诞树'})

# 退出登录
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')