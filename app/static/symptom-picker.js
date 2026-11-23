/* Symptom picker: type-ahead over /api/symptoms (full OptimusKG phenotype list plus the
   everyday-symptom list), removable chips, and one-tap common symptoms.
   Markup: <div class="sp" data-name="symptoms" data-initial='[{"id":..,"name":..}]'> ... </div>
   Selected items are written as JSON to <input type="hidden" name="{data-name}">. */
(function () {
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }

  function init(root) {
    var name = root.dataset.name || 'symptoms';
    var plain = root.dataset.plain === '1';            // patients: prefer plain labels
    var hidden = root.querySelector('input[type=hidden]');
    if (!hidden) { hidden = document.createElement('input'); hidden.type = 'hidden'; hidden.name = name; root.appendChild(hidden); }
    var chips = root.querySelector('.sp-chips');
    var input = root.querySelector('.sp-input');
    var box = root.querySelector('.sp-results');
    var selected = [];
    try { selected = JSON.parse(root.dataset.initial || '[]') || []; } catch (e) { selected = []; }
    var results = [], active = -1, timer = null, lastQ = '';

    function has(id) { return selected.some(function (s) { return s.id === id; }); }
    function sync() {
      hidden.value = JSON.stringify(selected);
      chips.innerHTML = selected.map(function (s, i) {
        return '<span class="sp-chip" title="' + esc(s.id) + '">' + esc(s.label || s.name) +
          '<button type="button" data-i="' + i + '" aria-label="Remove">×</button></span>';
      }).join('');
      root.querySelectorAll('.sp-common button[data-id]').forEach(function (b) {
        b.classList.toggle('on', has(b.dataset.id));
      });
      root.dispatchEvent(new CustomEvent('sp-change', {detail: selected, bubbles: true}));
    }
    function add(item) {
      if (!has(item.id)) selected.push({id: item.id, name: item.name, label: item.label || ''});
      sync();
    }
    function remove(id) { selected = selected.filter(function (s) { return s.id !== id; }); sync(); }

    chips.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-i]');
      if (b) { selected.splice(+b.dataset.i, 1); sync(); }
    });
    root.querySelectorAll('.sp-common button[data-id]').forEach(function (b) {
      b.addEventListener('click', function () {
        if (has(b.dataset.id)) remove(b.dataset.id);
        else add({id: b.dataset.id, name: b.dataset.name, label: plain ? b.dataset.label : ''});
      });
    });

    function render() {
      if (!results.length) {
        box.innerHTML = lastQ.length >= 2 ? '<div class="sp-empty">No match. Press Enter to add “' + esc(lastQ) + '” as typed.</div>' : '';
        box.hidden = lastQ.length < 2;
        return;
      }
      box.hidden = false;
      box.innerHTML = results.map(function (r, i) {
        var title = plain && r.label ? r.label : r.name;
        var sub = [];
        if (r.label && !plain && r.label.toLowerCase() !== r.name.toLowerCase()) sub.push(r.label);
        if (r.matched && r.matched !== r.name && r.matched !== r.label) sub.push('matches “' + r.matched + '”');
        sub.push(r.category);
        return '<div class="sp-opt' + (i === active ? ' active' : '') + (has(r.id) ? ' picked' : '') + '" data-i="' + i + '">' +
          '<div><b>' + esc(title) + '</b>' + (r.common ? ' <span class="tag completed">common</span>' : '') +
          '<div class="small muted">' + esc(sub.join(' · ')) + '</div></div>' +
          (plain ? '' : '<span class="small muted">' + esc(r.id) + '</span>') + '</div>';
      }).join('');
    }
    function search(q) {
      lastQ = q;
      if (q.length < 2) { results = []; render(); return; }
      fetch('/api/symptoms?limit=12&q=' + encodeURIComponent(q), {credentials: 'same-origin'})
        .then(function (r) { return r.ok ? r.json() : {items: []}; })
        .then(function (d) { if (q === lastQ) { results = d.items || []; active = results.length ? 0 : -1; render(); } })
        .catch(function () {});
    }
    input.addEventListener('input', function () {
      clearTimeout(timer);
      var q = input.value.trim();
      timer = setTimeout(function () { search(q); }, 140);
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { active = Math.min(active + 1, results.length - 1); render(); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { active = Math.max(active - 1, 0); render(); e.preventDefault(); }
      else if (e.key === 'Enter') {
        e.preventDefault();
        var q = input.value.trim();
        if (active >= 0 && results[active]) add({id: results[active].id, name: results[active].name, label: plain ? results[active].label : ''});
        else if (q.length >= 2) add({id: 'text:' + q.toLowerCase(), name: q});
        input.value = ''; results = []; lastQ = ''; render();
      } else if (e.key === 'Escape') { box.hidden = true; }
    });
    box.addEventListener('mousedown', function (e) {
      var o = e.target.closest('.sp-opt');
      if (!o) return;
      e.preventDefault();
      var r = results[+o.dataset.i];
      add({id: r.id, name: r.name, label: plain ? r.label : ''});
      input.value = ''; results = []; lastQ = ''; render(); input.focus();
    });
    input.addEventListener('blur', function () { setTimeout(function () { box.hidden = true; }, 150); });
    input.addEventListener('focus', function () { if (results.length) box.hidden = false; });
    sync();
  }
  document.querySelectorAll('.sp').forEach(init);
})();
