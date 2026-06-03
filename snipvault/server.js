const express = require('express');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const db = require('./db');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

app.use(session({
  secret: 'snipvault-secret-2024-xk9',
  resave: false,
  saveUninitialized: false,
  cookie: { maxAge: 7 * 24 * 60 * 60 * 1000 }
}));

const requireAuth = (req, res, next) => {
  if (!req.session.userId) return res.status(401).json({ error: 'Unauthorized' });
  next();
};

const getUser = (req) => {
  if (!req.session.userId) return null;
  return db.prepare('SELECT id, username, email, bio, avatar_color, created_at FROM users WHERE id = ?').get(req.session.userId);
};

// ── Auth ──────────────────────────────────────────────────────────────────────

app.post('/api/auth/register', (req, res) => {
  const { username, email, password } = req.body;
  if (!username || !email || !password) return res.status(400).json({ error: 'All fields required' });
  if (username.length < 3) return res.status(400).json({ error: 'Username must be at least 3 characters' });
  if (password.length < 6) return res.status(400).json({ error: 'Password must be at least 6 characters' });
  if (!/^[a-zA-Z0-9_-]+$/.test(username)) return res.status(400).json({ error: 'Username can only contain letters, numbers, _ and -' });

  const existing = db.prepare('SELECT id FROM users WHERE email = ? OR username = ?').get(email, username);
  if (existing) return res.status(400).json({ error: 'Username or email already taken' });

  const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#ef4444'];
  const hash = bcrypt.hashSync(password, 10);
  const id = uuidv4();
  const color = colors[Math.floor(Math.random() * colors.length)];

  db.prepare('INSERT INTO users (id, username, email, password_hash, avatar_color) VALUES (?, ?, ?, ?, ?)')
    .run(id, username, email, hash, color);

  req.session.userId = id;
  res.json({ success: true, username });
});

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body;
  const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email);
  if (!user || !bcrypt.compareSync(password, user.password_hash))
    return res.status(401).json({ error: 'Invalid email or password' });

  req.session.userId = user.id;
  res.json({ success: true, username: user.username });
});

app.post('/api/auth/logout', (req, res) => {
  req.session.destroy();
  res.json({ success: true });
});

app.get('/api/auth/me', (req, res) => {
  const user = getUser(req);
  if (!user) return res.json({ user: null });
  const snippetCount = db.prepare('SELECT COUNT(*) as c FROM snippets WHERE user_id = ?').get(user.id).c;
  const starCount = db.prepare('SELECT COUNT(*) as c FROM stars WHERE user_id = ?').get(user.id).c;
  const followerCount = db.prepare('SELECT COUNT(*) as c FROM follows WHERE following_id = ?').get(user.id).c;
  res.json({ user: { ...user, snippetCount, starCount, followerCount } });
});

// ── Snippets ──────────────────────────────────────────────────────────────────

app.get('/api/snippets', (req, res) => {
  const { lang, tag, q, sort = 'new', page = 1 } = req.query;
  const limit = 12;
  const offset = (page - 1) * limit;
  const userId = req.session.userId;

  let where = 's.is_public = 1';
  const params = [];

  if (lang) { where += ' AND s.language = ?'; params.push(lang); }
  if (tag) { where += ` AND s.tags LIKE ?`; params.push(`%"${tag}"%`); }
  if (q) { where += ' AND (s.title LIKE ? OR s.description LIKE ? OR u.username LIKE ?)'; params.push(`%${q}%`, `%${q}%`, `%${q}%`); }

  const orderMap = { new: 's.created_at DESC', stars: 's.stars DESC', views: 's.views DESC' };
  const order = orderMap[sort] || orderMap.new;

  const snippets = db.prepare(`
    SELECT s.*, u.username, u.avatar_color,
      ${userId ? '(SELECT 1 FROM stars WHERE user_id = ? AND snippet_id = s.id) as starred' : '0 as starred'}
    FROM snippets s JOIN users u ON s.user_id = u.id
    WHERE ${where}
    ORDER BY ${order} LIMIT ? OFFSET ?
  `).all(...(userId ? [userId] : []), ...params, limit, offset);

  const total = db.prepare(`
    SELECT COUNT(*) as c FROM snippets s JOIN users u ON s.user_id = u.id WHERE ${where}
  `).get(...params).c;

  res.json({ snippets: snippets.map(parseSnippet), total, pages: Math.ceil(total / limit) });
});

app.get('/api/snippets/:id', (req, res) => {
  const userId = req.session.userId;
  const snippet = db.prepare(`
    SELECT s.*, u.username, u.avatar_color, u.bio,
      ${userId ? '(SELECT 1 FROM stars WHERE user_id = ? AND snippet_id = s.id) as starred' : '0 as starred'}
    FROM snippets s JOIN users u ON s.user_id = u.id
    WHERE s.id = ? AND (s.is_public = 1 OR s.user_id = ?)
  `).get(...(userId ? [userId] : []), req.params.id, userId || 'none');

  if (!snippet) return res.status(404).json({ error: 'Snippet not found' });

  if (!userId || userId !== snippet.user_id) {
    db.prepare('UPDATE snippets SET views = views + 1 WHERE id = ?').run(snippet.id);
    snippet.views++;
  }

  res.json(parseSnippet(snippet));
});

app.post('/api/snippets', requireAuth, (req, res) => {
  const { title, description, code, language, tags, is_public } = req.body;
  if (!title || !code) return res.status(400).json({ error: 'Title and code are required' });

  const id = uuidv4();
  db.prepare(`INSERT INTO snippets (id, user_id, title, description, code, language, tags, is_public)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)`).run(
    id, req.session.userId, title.trim(), description || '',
    code, language || 'plaintext',
    JSON.stringify(Array.isArray(tags) ? tags : []),
    is_public ? 1 : 0
  );

  res.json({ id });
});

app.put('/api/snippets/:id', requireAuth, (req, res) => {
  const snippet = db.prepare('SELECT * FROM snippets WHERE id = ? AND user_id = ?').get(req.params.id, req.session.userId);
  if (!snippet) return res.status(404).json({ error: 'Not found' });

  const { title, description, code, language, tags, is_public } = req.body;
  db.prepare(`UPDATE snippets SET title=?, description=?, code=?, language=?, tags=?, is_public=?, updated_at=CURRENT_TIMESTAMP
    WHERE id=?`).run(
    title || snippet.title, description ?? snippet.description,
    code || snippet.code, language || snippet.language,
    JSON.stringify(Array.isArray(tags) ? tags : JSON.parse(snippet.tags)),
    is_public !== undefined ? (is_public ? 1 : 0) : snippet.is_public,
    snippet.id
  );

  res.json({ success: true });
});

app.delete('/api/snippets/:id', requireAuth, (req, res) => {
  const result = db.prepare('DELETE FROM snippets WHERE id = ? AND user_id = ?').run(req.params.id, req.session.userId);
  if (!result.changes) return res.status(404).json({ error: 'Not found' });
  res.json({ success: true });
});

// ── Stars ─────────────────────────────────────────────────────────────────────

app.post('/api/snippets/:id/star', requireAuth, (req, res) => {
  const snippet = db.prepare('SELECT id, user_id FROM snippets WHERE id = ?').get(req.params.id);
  if (!snippet) return res.status(404).json({ error: 'Not found' });

  const existing = db.prepare('SELECT 1 FROM stars WHERE user_id = ? AND snippet_id = ?').get(req.session.userId, snippet.id);
  if (existing) {
    db.prepare('DELETE FROM stars WHERE user_id = ? AND snippet_id = ?').run(req.session.userId, snippet.id);
    db.prepare('UPDATE snippets SET stars = stars - 1 WHERE id = ?').run(snippet.id);
    res.json({ starred: false });
  } else {
    db.prepare('INSERT INTO stars (user_id, snippet_id) VALUES (?, ?)').run(req.session.userId, snippet.id);
    db.prepare('UPDATE snippets SET stars = stars + 1 WHERE id = ?').run(snippet.id);
    res.json({ starred: true });
  }
});

// ── Users ─────────────────────────────────────────────────────────────────────

app.get('/api/users/:username', (req, res) => {
  const user = db.prepare('SELECT id, username, bio, avatar_color, created_at FROM users WHERE username = ?').get(req.params.username);
  if (!user) return res.status(404).json({ error: 'User not found' });

  const isOwner = req.session.userId === user.id;
  const snippets = db.prepare(`
    SELECT s.*, u.username, u.avatar_color FROM snippets s JOIN users u ON s.user_id = u.id
    WHERE s.user_id = ? ${isOwner ? '' : 'AND s.is_public = 1'}
    ORDER BY s.created_at DESC
  `).all(user.id);

  const followerCount = db.prepare('SELECT COUNT(*) as c FROM follows WHERE following_id = ?').get(user.id).c;
  const followingCount = db.prepare('SELECT COUNT(*) as c FROM follows WHERE follower_id = ?').get(user.id).c;
  const isFollowing = req.session.userId
    ? !!db.prepare('SELECT 1 FROM stars WHERE user_id = ? AND snippet_id IN (SELECT id FROM snippets WHERE user_id = ?)').get(req.session.userId, user.id)
    : false;

  res.json({ user: { ...user, followerCount, followingCount }, snippets: snippets.map(parseSnippet), isOwner });
});

app.put('/api/users/me', requireAuth, (req, res) => {
  const { bio } = req.body;
  db.prepare('UPDATE users SET bio = ? WHERE id = ?').run(bio || '', req.session.userId);
  res.json({ success: true });
});

app.get('/api/stats', (req, res) => {
  const users = db.prepare('SELECT COUNT(*) as c FROM users').get().c;
  const snippets = db.prepare('SELECT COUNT(*) as c FROM snippets WHERE is_public = 1').get().c;
  const stars = db.prepare('SELECT SUM(stars) as c FROM snippets').get().c || 0;
  const langs = db.prepare('SELECT language, COUNT(*) as count FROM snippets WHERE is_public=1 GROUP BY language ORDER BY count DESC LIMIT 8').all();
  res.json({ users, snippets, stars, langs });
});

// ── SPA fallback ──────────────────────────────────────────────────────────────

app.get('/{*path}', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

function parseSnippet(s) {
  try { s.tags = JSON.parse(s.tags); } catch { s.tags = []; }
  s.starred = !!s.starred;
  return s;
}

app.listen(PORT, () => console.log(`SnipVault running on http://localhost:${PORT}`));
