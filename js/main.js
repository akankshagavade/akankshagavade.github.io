function applyMode(mode){
  document.body.classList.toggle('plain', mode === 'plain');
  document.querySelectorAll('.mode-switch .lbl').forEach(function(l){ l.classList.remove('on'); });
  var activeSel = mode === 'plain' ? '.mode-switch .pro' : '.mode-switch .fun';
  var el = document.querySelector(activeSel);
  if(el) el.classList.add('on');
}

function toggleMode(){
  var next = document.body.classList.contains('plain') ? 'fun' : 'plain';
  try { localStorage.setItem('site-mode', next); } catch(e){}
  applyMode(next);
}

function openModal(id){
  var modal = document.getElementById('modal-' + id);
  if(!modal) return;
  modal.classList.add('active');
  document.getElementById('scrim').classList.add('active');
}

function closeModal(){
  document.querySelectorAll('.modal').forEach(function(m){ m.classList.remove('active'); });
  document.getElementById('scrim').classList.remove('active');
}

document.addEventListener('keydown', function(e){ if(e.key === 'Escape') closeModal(); });

document.addEventListener('DOMContentLoaded', function(){
  var saved = 'fun';
  try { saved = localStorage.getItem('site-mode') || 'fun'; } catch(e){}
  applyMode(saved);

  var page = document.body.getAttribute('data-page');
  document.querySelectorAll('.folder-btn').forEach(function(b){
    b.classList.toggle('is-active', b.dataset.page === page);
  });
  document.querySelectorAll('.plain-nav a.nav-link').forEach(function(a){
    a.classList.toggle('is-active', a.dataset.page === page);
  });
});
