/* Clinical images: lightbox, drop zones, and the image selector on the AI diagnosis page. */
(function () {
  // ---- lightbox
  var lb = document.getElementById('lightbox');
  document.addEventListener('click', function (e) {
    var t = e.target.closest('.img-thumb');
    if (!t || !lb) return;
    e.preventDefault();
    var full = lb.querySelector('img');
    full.removeAttribute('src');          // never flash the previous image
    full.src = t.dataset.full;
    lb.querySelector('.lb-caption').textContent = t.dataset.caption || '';
    lb.querySelector('.lb-open').href = t.dataset.full;
    if (lb.showModal) lb.showModal(); else window.open(t.dataset.full);
  });
  if (lb) lb.addEventListener('click', function (e) { if (e.target === lb) lb.close(); });

  // ---- drop zones (show chosen file names, accept drag and drop)
  document.querySelectorAll('[data-dropzone]').forEach(function (dz) {
    var input = dz.querySelector('input[type=file]'), out = dz.querySelector('.dz-files');
    function show() {
      var n = input.files.length;
      if (out) out.textContent = n ? (n === 1 ? input.files[0].name : n + ' files selected') : '';
    }
    input.addEventListener('change', show);
    ['dragenter', 'dragover'].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add('over'); });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove('over'); });
    });
    dz.addEventListener('drop', function (e) {
      if (e.dataTransfer && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show();
        input.dispatchEvent(new Event('change', { bubbles: true })); }
    });
  });

  // ---- AI page: pick up to N images; upload new ones without leaving the page
  var picker = document.querySelector('[data-image-picker]');
  if (!picker) return;
  var max = parseInt(picker.dataset.max || '4', 10);
  var list = picker.querySelector('.ip-list'), count = picker.querySelector('.ip-count');
  var file = picker.querySelector('.ip-file'), status = picker.querySelector('.ip-status');
  var tpl = picker.querySelector('template');

  function refresh() {
    var boxes = list.querySelectorAll('input[name=image_ids]');
    var on = Array.prototype.filter.call(boxes, function (b) { return b.checked; }).length;
    boxes.forEach(function (b) {
      b.disabled = !b.checked && on >= max;
      b.closest('.ip-item').classList.toggle('on', b.checked);
      b.closest('.ip-item').querySelectorAll('textarea,select').forEach(function (el) { el.disabled = !b.checked; });
    });
    if (count) count.textContent = on + ' of ' + max + ' selected';
    var empty = picker.querySelector('.ip-empty');
    if (empty) empty.hidden = boxes.length > 0;
  }
  list.addEventListener('change', refresh);
  refresh();

  function addItem(img) {
    if (list.querySelector('[data-id="' + img.id + '"]')) {
      var ex = list.querySelector('[data-id="' + img.id + '"] input[name=image_ids]');
      if (!ex.checked && !ex.disabled) ex.checked = true;
      return refresh();
    }
    var node = tpl.content.firstElementChild.cloneNode(true);
    node.dataset.id = img.id;
    node.querySelector('img').src = img.thumb;
    node.querySelector('.img-thumb').dataset.full = img.url;
    var cb = node.querySelector('input[name=image_ids]');
    cb.value = img.id;
    var on = list.querySelectorAll('input[name=image_ids]:checked').length;
    cb.checked = on < max;
    var ta = node.querySelector('textarea');
    ta.name = 'finding_' + img.id; ta.value = img.finding || '';
    var sel = node.querySelector('select');
    sel.name = 'modality_' + img.id; sel.value = img.modality;
    node.querySelector('.ip-label').textContent = img.modality_label + (img.body_site ? ' · ' + img.body_site : '');
    list.insertBefore(node, list.firstChild);
    refresh();
  }

  if (file) file.addEventListener('change', function () {
    if (!file.files.length) return;
    var fd = new FormData();
    Array.prototype.forEach.call(file.files, function (f) { fd.append('files', f); });
    fd.append('modality', picker.querySelector('.ip-modality').value);
    status.textContent = 'Uploading…';
    fetch(picker.dataset.upload + '?format=json', { method: 'POST', body: fd, credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        (data.images || []).forEach(addItem);
        status.textContent = (data.errors || []).join(' ') ||
          ((data.images || []).length ? 'Added to the patient record. Describe what each image shows.' : '');
        file.value = '';
        var first = list.querySelector('.ip-item.on textarea');
        if (first && !first.value) first.focus();
      })
      .catch(function () { status.textContent = 'Upload failed. Check the connection and try again.'; });
  });
})();
