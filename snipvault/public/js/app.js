/* ── State ───────────────────────────────────────────────── */
let currentUser = null;
let currentPage = '/';
let currentSnippetTags = [];

const LANGUAGES = [
  'plaintext','javascript','typescript','python','go','rust','java','kotlin','swift',
  'c','cpp','csharp','php','ruby','bash','sql','html','css','json','yaml','markdown',
  'dockerfile','lua','scala','haskell','elixir','clojure','r','dart','zig'
];

const LANG_ICON = {
  javascript:'🟨', typescript:'🔷', python:'🐍', go:'🐹', rust:'🦀',
  java:'☕', kotlin:'🎯', swift:'🍎', bash:'💻', sql:'🗄️', html:'🌐',
  css:'🎨', json:'📋', rust:'🦀', ruby:'💎', php:'🐘', docker:'🐋',
  dockerfile:'🐋', markdown:'📝', plaintext:'📄',
};

/* ── Router ──────────────────────────────────────────────── */
window.addEventListener('popstate', () => renderPage(location.pathname));
function navigate(path) {
  history.pushState({}, '', path);
  renderPage(path);
  window.scrollTo(0, 0);
}

async function renderPage(path) {
  currentPage = path;
  const main = document.getElementById('main');

  // Match routes
  if (path === '/') return renderHome(main);
  if (path === '/explore') return renderExplore(main);
  if (path === '/login') return renderLogin(main);
  if (path === '/register') return renderRegister(main);
  if (path === '/new') return renderNew(main);
  if (path.startsWith('/snippet/')) return renderSnippetDetail(main, path.slice(9));
  if (path.startsWith('/user/')) return renderProfile(main, path.slice(6));
  if (path === '/dashboard') return renderDashboard(main);

  main.innerHTML = `<div class="container page"><div class="empty"><div class="empty-icon">🔍</div><h3>Page not found</h3><p>The page you're looking for doesn't exist.</p><a href="/" onclick="navigate('/');return false;" class="btn btn-primary">Go Home</a></div></div>`;
}

/* ── API ─────────────────────────────────────────────────── */
async function api(method, path, body) {
  const res = await fetch('/api' + path, {
    method, credentials: 'include',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

/* ── Auth helpers ────────────────────────────────────────── */
async function loadUser() {
  try {
    const data = await api('GET', '/auth/me');
    currentUser = data.user;
  } catch { currentUser = null; }
  renderNav();
}

function renderNav() {
  const el = document.getElementById('nav-auth');
  if (currentUser) {
    el.innerHTML = `
      <a href="/new" onclick="navigate('/new');return false;" class="btn btn-primary btn-sm">+ New Snippet</a>
      <div class="nav-user-menu">
        <a href="/dashboard" onclick="navigate('/dashboard');return false;" style="display:flex;align-items:center;gap:6px;">
          <div class="nav-avatar" style="background:${currentUser.avatar_color}">${currentUser.username[0].toUpperCase()}</div>
        </a>
      </div>
    `;
  } else {
    el.innerHTML = `
      <a href="/login" onclick="navigate('/login');return false;" class="btn btn-ghost btn-sm">Log in</a>
      <a href="/register" onclick="navigate('/register');return false;" class="btn btn-primary btn-sm">Sign up free</a>
    `;
  }
}

/* ── Toast ───────────────────────────────────────────────── */
function toast(msg, type = 'success') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.innerHTML = `<span>${type === 'success' ? '✓' : '✕'}</span> ${msg}`;
  c.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

/* ── Modal ───────────────────────────────────────────────── */
function openModal(html) {
  document.getElementById('modal-content').innerHTML = html;
  document.getElementById('modal-overlay').classList.remove('hidden');
}
function closeModal() {
  document.getElementById('modal-overlay').classList.add('hidden');
}
document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay')) closeModal();
});

/* ── Pages ───────────────────────────────────────────────── */
async function renderHome(main) {
  const stats = await api('GET', '/stats').catch(() => ({ users: 0, snippets: 0, stars: 0, langs: [] }));
  const { snippets } = await api('GET', '/snippets?sort=stars&page=1').catch(() => ({ snippets: [] }));

  main.innerHTML = `
    <div class="hero">
      <div class="hero-badge"><div class="hero-badge-dot"></div> The developer's code library</div>
      <h1>Store, Share &amp; Discover<br><span class="gradient">Code Snippets</span></h1>
      <p>Save your best code snippets, share them with the world, and discover amazing code from other developers.</p>
      <div class="hero-actions">
        ${currentUser
          ? `<a href="/new" onclick="navigate('/new');return false;" class="btn btn-primary" style="font-size:1rem;padding:12px 28px;">+ Create Snippet</a>
             <a href="/explore" onclick="navigate('/explore');return false;" class="btn btn-ghost" style="font-size:1rem;padding:12px 28px;">Explore →</a>`
          : `<a href="/register" onclick="navigate('/register');return false;" class="btn btn-primary" style="font-size:1rem;padding:12px 28px;">Get started free</a>
             <a href="/explore" onclick="navigate('/explore');return false;" class="btn btn-ghost" style="font-size:1rem;padding:12px 28px;">Explore snippets →</a>`
        }
      </div>
      <div class="hero-stats">
        <div><div class="hero-stat-value">${fmtNum(stats.snippets)}</div><div class="hero-stat-label">Snippets shared</div></div>
        <div><div class="hero-stat-value">${fmtNum(stats.users)}</div><div class="hero-stat-label">Developers</div></div>
        <div><div class="hero-stat-value">${fmtNum(stats.stars)}</div><div class="hero-stat-label">Stars given</div></div>
      </div>
    </div>

    <div class="features">
      ${[
        ['🎨', 'Syntax Highlighting', 'Beautiful code highlighting for 30+ languages powered by highlight.js.', '#6366f1'],
        ['🔒', 'Public or Private', 'Choose to share your snippets publicly or keep them private.', '#8b5cf6'],
        ['⭐', 'Star Favourites', 'Star snippets you love and build your personal collection.', '#f59e0b'],
        ['🔍', 'Powerful Search', 'Find any snippet instantly by title, description, author or language.', '#10b981'],
        ['🏷️', 'Tags & Filters', 'Organise with tags and filter by language to find exactly what you need.', '#ec4899'],
        ['📊', 'View Analytics', 'See how many developers are reading and starring your snippets.', '#3b82f6'],
      ].map(([icon, title, desc, color]) => `
        <div class="feature-card">
          <div class="feature-icon" style="background:${color}20;color:${color}">${icon}</div>
          <div class="feature-title">${title}</div>
          <div class="feature-desc">${desc}</div>
        </div>
      `).join('')}
    </div>

    ${snippets.length ? `
    <div class="container" style="padding-bottom:64px;">
      <div class="section-title">
        🌟 Top Snippets
        <a href="/explore" onclick="navigate('/explore');return false;" class="btn btn-ghost btn-sm">View all →</a>
      </div>
      <div class="grid">${snippets.slice(0, 6).map(snippetCard).join('')}</div>
    </div>` : ''}
  `;
  attachSnippetClicks(main);
  attachStarClicks(main);
}

async function renderExplore(main) {
  main.innerHTML = `<div class="container page"><div class="section-title">Explore Snippets</div><div id="explore-content"><div class="empty"><div class="empty-icon">⏳</div><p>Loading...</p></div></div></div>`;
  await loadExplore(1, {});
}

async function loadExplore(page, filters) {
  const params = new URLSearchParams({ page, ...filters });
  const { snippets, total, pages } = await api('GET', `/snippets?${params}`).catch(() => ({ snippets: [], total: 0, pages: 0 }));

  const container = document.getElementById('explore-content');
  if (!container) return;

  container.innerHTML = `
    <div class="filter-bar">
      <div class="search-wrap">
        <span class="search-icon">🔍</span>
        <input type="text" id="search-input" placeholder="Search snippets…" value="${filters.q || ''}">
      </div>
      <select class="filter-select" id="lang-filter">
        <option value="">All languages</option>
        ${LANGUAGES.map(l => `<option value="${l}" ${filters.lang === l ? 'selected' : ''}>${l}</option>`).join('')}
      </select>
      <div class="tab-group">
        ${['new','stars','views'].map(s => `
          <button class="tab-btn ${(filters.sort || 'new') === s ? 'active' : ''}" data-sort="${s}">${s === 'new' ? '🕐 Newest' : s === 'stars' ? '⭐ Top' : '👁️ Trending'}</button>
        `).join('')}
      </div>
    </div>

    <div style="color:var(--text2);font-size:.88rem;margin-bottom:16px;">${total} snippet${total !== 1 ? 's' : ''} found</div>

    ${snippets.length
      ? `<div class="grid">${snippets.map(snippetCard).join('')}</div>`
      : `<div class="empty"><div class="empty-icon">🔍</div><h3>No snippets found</h3><p>Try different search terms or filters.</p></div>`
    }

    ${pages > 1 ? `<div class="pagination">${Array.from({length: Math.min(pages, 7)}, (_, i) => {
      const p = i + 1;
      return `<button class="page-btn ${p === page ? 'active' : ''}" data-page="${p}">${p}</button>`;
    }).join('')}</div>` : ''}
  `;

  // Events
  let searchTimer;
  container.querySelector('#search-input')?.addEventListener('input', e => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadExplore(1, { ...filters, q: e.target.value }), 400);
  });
  container.querySelector('#lang-filter')?.addEventListener('change', e => {
    loadExplore(1, { ...filters, lang: e.target.value });
  });
  container.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => loadExplore(1, { ...filters, sort: btn.dataset.sort }));
  });
  container.querySelectorAll('.page-btn').forEach(btn => {
    btn.addEventListener('click', () => loadExplore(parseInt(btn.dataset.page), filters));
  });
  attachSnippetClicks(container);
  attachStarClicks(container);
}

async function renderSnippetDetail(main, id) {
  main.innerHTML = `<div class="container page"><div class="empty"><div class="empty-icon">⏳</div><p>Loading…</p></div></div>`;
  const snippet = await api('GET', `/snippets/${id}`).catch(() => null);
  if (!snippet) {
    main.innerHTML = `<div class="container page"><div class="empty"><div class="empty-icon">😕</div><h3>Snippet not found</h3><a href="/explore" onclick="navigate('/explore');return false;" class="btn btn-primary mt-4">Browse snippets</a></div></div>`;
    return;
  }

  const isOwner = currentUser && currentUser.id === snippet.user_id;

  main.innerHTML = `
    <div class="container page">
      <div class="snippet-detail">
        <div style="margin-bottom:20px;">
          <a href="/explore" onclick="navigate('/explore');return false;" class="btn btn-ghost btn-sm">← Explore</a>
        </div>
        <div class="detail-header">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <span class="lang-badge">${snippet.language}</span>
            ${snippet.is_public ? '<span class="lang-badge" style="color:var(--green);border-color:var(--green);background:rgba(16,185,129,0.1);">🌍 Public</span>' : '<span class="lang-badge" style="color:var(--text3);">🔒 Private</span>'}
            ${snippet.tags.map(t => `<span class="tag-chip">#${t}</span>`).join('')}
          </div>
          <h1 class="detail-title">${escHtml(snippet.title)}</h1>
          ${snippet.description ? `<p style="color:var(--text2);margin-bottom:12px;">${escHtml(snippet.description)}</p>` : ''}
          <div class="detail-meta">
            <a href="/user/${snippet.username}" onclick="navigate('/user/${snippet.username}');return false;" style="display:flex;align-items:center;gap:6px;">
              <div class="avatar-sm" style="background:${snippet.avatar_color}">${snippet.username[0].toUpperCase()}</div>
              <strong>${snippet.username}</strong>
            </a>
            <span>·</span>
            <span>${timeAgo(snippet.created_at)}</span>
            <span>·</span>
            <span>👁️ ${snippet.views} views</span>
            <span>·</span>
            <span>⭐ ${snippet.stars} stars</span>
          </div>
          <div class="detail-actions">
            ${currentUser ? `
              <button class="btn btn-ghost btn-sm star-btn" data-id="${snippet.id}" data-starred="${snippet.starred}">
                ${snippet.starred ? '⭐ Starred' : '☆ Star'}
              </button>
            ` : `<a href="/login" onclick="navigate('/login');return false;" class="btn btn-ghost btn-sm">☆ Star</a>`}
            <button class="btn btn-ghost btn-sm" onclick="copyCode()">📋 Copy</button>
            ${isOwner ? `
              <a href="/edit/${snippet.id}" onclick="navigate('/edit/${snippet.id}');return false;" class="btn btn-ghost btn-sm">✏️ Edit</a>
              <button class="btn btn-ghost btn-sm" id="delete-btn" style="color:var(--red);">🗑️ Delete</button>
            ` : ''}
          </div>
        </div>

        <div class="code-block">
          <div class="code-block-header">
            <span class="lang">${snippet.language}</span>
            <button class="copy-btn" id="copy-btn" onclick="copyCode()">📋 Copy code</button>
          </div>
          <pre><code class="language-${snippet.language}" id="snippet-code">${escHtml(snippet.code)}</code></pre>
        </div>
      </div>
    </div>
  `;

  hljs.highlightElement(document.getElementById('snippet-code'));
  window._snippetCode = snippet.code;

  const starBtn = main.querySelector('.star-btn');
  if (starBtn) {
    starBtn.addEventListener('click', async () => {
      if (!currentUser) { navigate('/login'); return; }
      try {
        const { starred } = await api('POST', `/snippets/${snippet.id}/star`);
        starBtn.dataset.starred = starred;
        starBtn.textContent = starred ? '⭐ Starred' : '☆ Star';
        toast(starred ? 'Added to stars!' : 'Removed from stars');
      } catch (e) { toast(e.message, 'error'); }
    });
  }

  const delBtn = main.querySelector('#delete-btn');
  if (delBtn) {
    delBtn.addEventListener('click', () => {
      openModal(`
        <h2>Delete snippet?</h2>
        <p style="color:var(--text2);margin-bottom:24px;">This action cannot be undone. The snippet will be permanently deleted.</p>
        <div style="display:flex;gap:12px;">
          <button class="btn btn-danger" id="confirm-delete">Delete</button>
          <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
        </div>
      `);
      document.getElementById('confirm-delete').addEventListener('click', async () => {
        try {
          await api('DELETE', `/snippets/${snippet.id}`);
          closeModal();
          toast('Snippet deleted');
          navigate('/dashboard');
        } catch (e) { toast(e.message, 'error'); }
      });
    });
  }
}

window.copyCode = function() {
  const code = window._snippetCode || document.getElementById('snippet-code')?.textContent;
  if (!code) return;
  navigator.clipboard.writeText(code).then(() => {
    toast('Code copied to clipboard!');
    const btn = document.getElementById('copy-btn');
    if (btn) { btn.textContent = '✓ Copied!'; btn.classList.add('copied'); setTimeout(() => { btn.textContent = '📋 Copy code'; btn.classList.remove('copied'); }, 2000); }
  });
};

async function renderNew(main, editSnippet = null) {
  if (!currentUser) { navigate('/login'); return; }

  currentSnippetTags = editSnippet ? [...editSnippet.tags] : [];
  const isEdit = !!editSnippet;

  main.innerHTML = `
    <div class="container page">
      <div class="new-page">
        <h1>${isEdit ? '✏️ Edit Snippet' : '+ New Snippet'}</h1>
        <form id="snippet-form">
          <div class="form-group">
            <label class="form-label">Title *</label>
            <input type="text" id="s-title" placeholder="Give your snippet a clear title…" value="${isEdit ? escHtml(editSnippet.title) : ''}" required maxlength="100">
          </div>
          <div class="form-group">
            <label class="form-label">Description</label>
            <textarea id="s-desc" placeholder="What does this snippet do?" rows="2">${isEdit ? escHtml(editSnippet.description) : ''}</textarea>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label class="form-label">Language</label>
              <select id="s-lang">
                ${LANGUAGES.map(l => `<option value="${l}" ${isEdit && editSnippet.language === l ? 'selected' : (!isEdit && l === 'javascript' ? 'selected' : '')}>${l}</option>`).join('')}
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">Tags <span style="color:var(--text3);font-weight:400;">(press Enter to add)</span></label>
              <input type="text" id="s-tag-input" placeholder="e.g. react, hooks, api">
              <div class="tags-display" id="tags-display">${currentSnippetTags.map(tagChip).join('')}</div>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Code *</label>
            <div class="editor-wrap">
              <div class="editor-toolbar">
                <span class="editor-lang-label" id="editor-lang-label">javascript</span>
                <button type="button" class="copy-btn" onclick="copyEditorCode()">📋 Copy</button>
              </div>
              <textarea id="code-textarea" placeholder="Paste or type your code here…" spellcheck="false">${isEdit ? escHtml(editSnippet.code) : ''}</textarea>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Visibility</label>
            <div class="toggle-wrap">
              <label class="toggle">
                <input type="checkbox" id="s-public" ${!isEdit || editSnippet.is_public ? 'checked' : ''}>
                <div class="toggle-slider"></div>
              </label>
              <span id="vis-label" style="color:var(--text2);font-size:.9rem;">🌍 Public — visible to everyone</span>
            </div>
          </div>
          <div id="form-error" class="form-error" style="display:none;"></div>
          <div style="display:flex;gap:12px;margin-top:8px;">
            <button type="submit" class="btn btn-primary">${isEdit ? 'Save changes' : '🚀 Publish Snippet'}</button>
            <button type="button" class="btn btn-ghost" onclick="history.back()">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  `;

  // Tag input
  const tagInput = document.getElementById('s-tag-input');
  tagInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const val = tagInput.value.trim().toLowerCase().replace(/[^a-z0-9-]/g, '');
      if (val && !currentSnippetTags.includes(val) && currentSnippetTags.length < 5) {
        currentSnippetTags.push(val);
        document.getElementById('tags-display').innerHTML = currentSnippetTags.map(tagChip).join('');
        tagInput.value = '';
        attachTagRemove();
      }
    }
  });
  attachTagRemove();

  // Lang label sync
  document.getElementById('s-lang').addEventListener('change', e => {
    document.getElementById('editor-lang-label').textContent = e.target.value;
  });

  // Visibility label
  document.getElementById('s-public').addEventListener('change', e => {
    document.getElementById('vis-label').textContent = e.target.checked
      ? '🌍 Public — visible to everyone'
      : '🔒 Private — only you can see this';
  });

  // Form submit
  document.getElementById('snippet-form').addEventListener('submit', async e => {
    e.preventDefault();
    const errEl = document.getElementById('form-error');
    errEl.style.display = 'none';
    const code = document.getElementById('code-textarea').value;
    if (!code.trim()) { errEl.textContent = 'Code cannot be empty.'; errEl.style.display = 'block'; return; }

    const body = {
      title: document.getElementById('s-title').value,
      description: document.getElementById('s-desc').value,
      code,
      language: document.getElementById('s-lang').value,
      tags: currentSnippetTags,
      is_public: document.getElementById('s-public').checked,
    };
    try {
      if (isEdit) {
        await api('PUT', `/snippets/${editSnippet.id}`, body);
        toast('Snippet updated!');
        navigate(`/snippet/${editSnippet.id}`);
      } else {
        const { id } = await api('POST', '/snippets', body);
        toast('Snippet published! 🚀');
        navigate(`/snippet/${id}`);
      }
    } catch (err) {
      errEl.textContent = err.message;
      errEl.style.display = 'block';
    }
  });
}

function attachTagRemove() {
  document.querySelectorAll('.tag-remove').forEach(el => {
    el.addEventListener('click', () => {
      const tag = el.dataset.tag;
      currentSnippetTags = currentSnippetTags.filter(t => t !== tag);
      document.getElementById('tags-display').innerHTML = currentSnippetTags.map(tagChip).join('');
      attachTagRemove();
    });
  });
}

window.copyEditorCode = function() {
  const code = document.getElementById('code-textarea').value;
  navigator.clipboard.writeText(code).then(() => toast('Copied!'));
};

async function renderLogin(main) {
  if (currentUser) { navigate('/'); return; }
  main.innerHTML = `
    <div class="auth-page">
      <div class="auth-card">
        <div class="auth-logo"><span class="logo-icon">⚡</span> <span class="logo-text">SnipVault</span></div>
        <h1 class="auth-title">Welcome back</h1>
        <p class="auth-sub">Sign in to your account to continue</p>
        <form id="login-form">
          <div class="form-group">
            <label class="form-label">Email</label>
            <input type="email" id="l-email" placeholder="you@example.com" required autocomplete="email">
          </div>
          <div class="form-group">
            <label class="form-label">Password</label>
            <input type="password" id="l-pass" placeholder="••••••••" required autocomplete="current-password">
          </div>
          <div id="login-error" class="form-error" style="display:none;"></div>
          <button type="submit" class="btn btn-primary w-full" style="margin-top:8px;justify-content:center;">Sign in</button>
        </form>
        <div class="auth-switch">Don't have an account? <a href="/register" onclick="navigate('/register');return false;">Sign up free</a></div>
      </div>
    </div>
  `;
  document.getElementById('login-form').addEventListener('submit', async e => {
    e.preventDefault();
    const errEl = document.getElementById('login-error');
    errEl.style.display = 'none';
    try {
      await api('POST', '/auth/login', { email: document.getElementById('l-email').value, password: document.getElementById('l-pass').value });
      await loadUser();
      toast(`Welcome back, ${currentUser.username}! 👋`);
      navigate('/dashboard');
    } catch (err) { errEl.textContent = err.message; errEl.style.display = 'block'; }
  });
}

async function renderRegister(main) {
  if (currentUser) { navigate('/'); return; }
  main.innerHTML = `
    <div class="auth-page">
      <div class="auth-card">
        <div class="auth-logo"><span class="logo-icon">⚡</span> <span class="logo-text">SnipVault</span></div>
        <h1 class="auth-title">Create your account</h1>
        <p class="auth-sub">Join thousands of developers sharing code</p>
        <form id="reg-form">
          <div class="form-group">
            <label class="form-label">Username</label>
            <input type="text" id="r-user" placeholder="cooldev42" required minlength="3" autocomplete="username">
          </div>
          <div class="form-group">
            <label class="form-label">Email</label>
            <input type="email" id="r-email" placeholder="you@example.com" required autocomplete="email">
          </div>
          <div class="form-group">
            <label class="form-label">Password</label>
            <input type="password" id="r-pass" placeholder="At least 6 characters" required minlength="6" autocomplete="new-password">
          </div>
          <div id="reg-error" class="form-error" style="display:none;"></div>
          <button type="submit" class="btn btn-primary w-full" style="margin-top:8px;justify-content:center;">Create account</button>
        </form>
        <div class="auth-switch">Already have an account? <a href="/login" onclick="navigate('/login');return false;">Sign in</a></div>
      </div>
    </div>
  `;
  document.getElementById('reg-form').addEventListener('submit', async e => {
    e.preventDefault();
    const errEl = document.getElementById('reg-error');
    errEl.style.display = 'none';
    try {
      await api('POST', '/auth/register', {
        username: document.getElementById('r-user').value,
        email: document.getElementById('r-email').value,
        password: document.getElementById('r-pass').value,
      });
      await loadUser();
      toast(`Welcome to SnipVault, ${currentUser.username}! 🎉`);
      navigate('/new');
    } catch (err) { errEl.textContent = err.message; errEl.style.display = 'block'; }
  });
}

async function renderDashboard(main) {
  if (!currentUser) { navigate('/login'); return; }
  main.innerHTML = `<div class="container page"><div class="empty"><div class="empty-icon">⏳</div><p>Loading dashboard…</p></div></div>`;

  const profile = await api('GET', `/users/${currentUser.username}`).catch(() => null);
  if (!profile) { main.innerHTML = `<div class="container page"><p>Error loading dashboard.</p></div>`; return; }

  const { user, snippets } = profile;
  const totalViews = snippets.reduce((a, s) => a + s.views, 0);
  const totalStars = snippets.reduce((a, s) => a + s.stars, 0);
  const pubCount = snippets.filter(s => s.is_public).length;

  main.innerHTML = `
    <div class="container page">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;flex-wrap:wrap;gap:12px;">
        <div>
          <h1 style="font-size:1.8rem;font-weight:800;">My Dashboard</h1>
          <p style="color:var(--text2);">Welcome back, ${user.username}!</p>
        </div>
        <div style="display:flex;gap:8px;">
          <button class="btn btn-ghost btn-sm" id="edit-profile-btn">✏️ Edit Profile</button>
          <button class="btn btn-ghost btn-sm" id="logout-btn" style="color:var(--red);">Sign out</button>
        </div>
      </div>

      <!-- Stats row -->
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px;margin-bottom:40px;">
        ${[
          ['📝', snippets.length, 'Snippets'],
          ['🌍', pubCount, 'Public'],
          ['👁️', fmtNum(totalViews), 'Total Views'],
          ['⭐', fmtNum(totalStars), 'Total Stars'],
        ].map(([icon, val, label]) => `
          <div class="card" style="padding:20px;text-align:center;">
            <div style="font-size:1.6rem;margin-bottom:8px;">${icon}</div>
            <div style="font-size:1.6rem;font-weight:800;">${val}</div>
            <div style="color:var(--text2);font-size:.85rem;">${label}</div>
          </div>
        `).join('')}
      </div>

      <!-- Snippets -->
      <div class="section-title">
        My Snippets
        <a href="/new" onclick="navigate('/new');return false;" class="btn btn-primary btn-sm">+ New</a>
      </div>
      ${snippets.length
        ? `<div class="grid">${snippets.map(s => snippetCard(s, true)).join('')}</div>`
        : `<div class="empty"><div class="empty-icon">✨</div><h3>No snippets yet</h3><p>Create your first snippet and share it with the world.</p><a href="/new" onclick="navigate('/new');return false;" class="btn btn-primary mt-4">Create first snippet</a></div>`
      }
    </div>
  `;

  attachSnippetClicks(main);
  attachStarClicks(main);

  document.getElementById('logout-btn').addEventListener('click', async () => {
    await api('POST', '/auth/logout');
    currentUser = null;
    renderNav();
    toast('Signed out. See you soon!');
    navigate('/');
  });

  document.getElementById('edit-profile-btn').addEventListener('click', () => {
    openModal(`
      <h2>Edit Profile</h2>
      <div class="form-group">
        <label class="form-label">Bio</label>
        <textarea id="edit-bio" rows="3" placeholder="Tell the world about yourself…">${escHtml(user.bio || '')}</textarea>
      </div>
      <div style="display:flex;gap:12px;margin-top:4px;">
        <button class="btn btn-primary" id="save-profile">Save</button>
        <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      </div>
    `);
    document.getElementById('save-profile').addEventListener('click', async () => {
      try {
        await api('PUT', '/users/me', { bio: document.getElementById('edit-bio').value });
        closeModal();
        toast('Profile updated!');
      } catch (e) { toast(e.message, 'error'); }
    });
  });
}

async function renderProfile(main, username) {
  main.innerHTML = `<div class="container page"><div class="empty"><div class="empty-icon">⏳</div><p>Loading profile…</p></div></div>`;
  const data = await api('GET', `/users/${username}`).catch(() => null);
  if (!data) { main.innerHTML = `<div class="container page"><div class="empty"><div class="empty-icon">😕</div><h3>User not found</h3></div></div>`; return; }

  const { user, snippets, isOwner } = data;
  main.innerHTML = `
    <div class="container page">
      <div class="profile-header">
        <div class="profile-avatar" style="background:${user.avatar_color}">${user.username[0].toUpperCase()}</div>
        <div class="profile-info">
          <div class="profile-name">@${user.username}</div>
          ${user.bio ? `<div class="profile-bio">${escHtml(user.bio)}</div>` : '<div class="profile-bio" style="color:var(--text3);font-style:italic;">No bio yet</div>'}
          <div class="profile-stats">
            <div class="profile-stat"><div class="profile-stat-value">${snippets.length}</div><div class="profile-stat-label">Snippets</div></div>
            <div class="profile-stat"><div class="profile-stat-value">${fmtNum(snippets.reduce((a,s)=>a+s.stars,0))}</div><div class="profile-stat-label">Stars earned</div></div>
            <div class="profile-stat"><div class="profile-stat-value">${fmtNum(snippets.reduce((a,s)=>a+s.views,0))}</div><div class="profile-stat-label">Views</div></div>
          </div>
          ${isOwner ? `<div style="margin-top:16px;"><a href="/dashboard" onclick="navigate('/dashboard');return false;" class="btn btn-ghost btn-sm">Dashboard →</a></div>` : ''}
        </div>
      </div>

      <div class="section-title">${isOwner ? 'All Snippets' : `${username}'s Snippets`}</div>
      ${snippets.length
        ? `<div class="grid">${snippets.map(snippetCard).join('')}</div>`
        : `<div class="empty"><div class="empty-icon">📭</div><h3>No public snippets</h3><p>This user hasn't shared any snippets yet.</p></div>`
      }
    </div>
  `;
  attachSnippetClicks(main);
  attachStarClicks(main);
}

// Edit route
window.addEventListener('popstate', () => {});
(function overrideNavigate() {
  const orig = navigate;
  window.navigate = async function(path) {
    if (path.startsWith('/edit/')) {
      const id = path.slice(6);
      const snippet = await api('GET', `/snippets/${id}`).catch(() => null);
      if (snippet && currentUser && snippet.user_id === currentUser.id) {
        history.pushState({}, '', path);
        currentPage = path;
        window.scrollTo(0, 0);
        return renderNew(document.getElementById('main'), snippet);
      }
    }
    return orig(path);
  };
})();

/* ── Components ──────────────────────────────────────────── */
function snippetCard(s, showPrivate = false) {
  const langColor = { javascript: '#f7df1e', typescript: '#3178c6', python: '#3776ab', go: '#00add8', rust: '#f74c00', css: '#2965f1', html: '#e34f26', bash: '#4eaa25', ruby: '#cc342d', java: '#ed8b00', sql: '#336791', php: '#8892be' };
  const lc = langColor[s.language] || '#6b7591';
  return `
    <div class="card snippet-card" data-id="${s.id}">
      <div class="code-preview">
        <pre><code class="language-${s.language}">${escHtml(s.code.slice(0, 500))}</code></pre>
      </div>
      <div class="card-header">
        <div style="flex:1;min-width:0;">
          <div class="snippet-title">${escHtml(s.title)}</div>
          ${s.description ? `<div class="snippet-desc">${escHtml(s.description)}</div>` : ''}
        </div>
      </div>
      <div class="card-footer">
        <div class="snippet-meta" style="gap:10px;">
          <span style="display:flex;align-items:center;gap:4px;font-size:.78rem;font-weight:600;color:${lc}">
            <span>●</span>${s.language}
          </span>
          <a href="/user/${s.username}" onclick="event.stopPropagation();navigate('/user/${s.username}');return false;" class="snippet-author">
            <div class="avatar-sm" style="background:${s.avatar_color}">${s.username[0].toUpperCase()}</div>
            ${s.username}
          </a>
          ${!s.is_public ? '<span style="font-size:.75rem;color:var(--text3);">🔒</span>' : ''}
        </div>
        <div style="display:flex;gap:10px;align-items:center;">
          <span class="stat-item">👁️ ${fmtNum(s.views)}</span>
          <span class="stat-item ${s.starred ? 'active' : ''} star-btn" data-id="${s.id}" data-starred="${s.starred}" style="cursor:pointer;">
            ${s.starred ? '⭐' : '☆'} ${fmtNum(s.stars)}
          </span>
        </div>
      </div>
    </div>
  `;
}

function tagChip(t) {
  return `<span class="tag-remove" data-tag="${t}">#${t} <span>✕</span></span>`;
}

function attachSnippetClicks(container) {
  container.querySelectorAll('.snippet-card').forEach(el => {
    el.addEventListener('click', e => {
      if (e.target.closest('a, .star-btn, button')) return;
      navigate('/snippet/' + el.dataset.id);
    });
  });
  // Highlight code previews
  container.querySelectorAll('.code-preview code').forEach(el => {
    try { hljs.highlightElement(el); } catch {}
  });
}

function attachStarClicks(container) {
  container.querySelectorAll('.star-btn[data-id]').forEach(el => {
    el.addEventListener('click', async e => {
      e.stopPropagation();
      if (!currentUser) { navigate('/login'); return; }
      try {
        const { starred } = await api('POST', `/snippets/${el.dataset.id}/star`);
        el.dataset.starred = starred;
        const stars = parseInt(el.textContent.match(/\d+/)?.[0] || '0');
        el.innerHTML = `${starred ? '⭐' : '☆'} ${fmtNum(starred ? stars + 1 : Math.max(0, stars - 1))}`;
        el.classList.toggle('active', starred);
      } catch {}
    });
  });
}

/* ── Helpers ─────────────────────────────────────────────── */
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function timeAgo(dt) {
  const diff = (Date.now() - new Date(dt + 'Z').getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
  if (diff < 2592000) return `${Math.floor(diff/86400)}d ago`;
  return new Date(dt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function fmtNum(n) {
  n = parseInt(n) || 0;
  if (n >= 1000) return (n/1000).toFixed(1) + 'k';
  return n;
}

/* ── Init ────────────────────────────────────────────────── */
(async () => {
  await loadUser();
  renderPage(location.pathname);
})();
