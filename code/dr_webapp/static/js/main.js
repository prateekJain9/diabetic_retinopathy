// ── Image preview for dropzone ──
const imageInput  = document.getElementById('imageInput');
const preview     = document.getElementById('preview');
const dropzone    = document.getElementById('dropzone');
const dropInner   = document.getElementById('dropzoneInner');

if (imageInput) {
  imageInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (preview) {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        if (dropInner) dropInner.style.display = 'none';
      }
    };
    reader.readAsDataURL(file);
  });
}

// ── Drag-and-drop highlight ──
if (dropzone) {
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--accent)';
    dropzone.style.background  = 'rgba(91,143,249,.04)';
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = '';
    dropzone.style.background  = '';
  });
  dropzone.addEventListener('drop', (e) => {
    dropzone.style.borderColor = '';
    dropzone.style.background  = '';
  });
}

// ── Auto-dismiss alerts ──
document.querySelectorAll('.alert').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity .5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 500);
  }, 4000);
});

// ── Submit button loading state ──
const submitBtn  = document.getElementById('submitBtn');
const uploadForm = document.getElementById('uploadForm');
if (uploadForm && submitBtn) {
  uploadForm.addEventListener('submit', () => {
    submitBtn.textContent = 'Analyzing…';
    submitBtn.disabled = true;
    submitBtn.style.opacity = '.7';
  });
}
