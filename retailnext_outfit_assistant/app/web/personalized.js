const statusText = document.getElementById('status-text');
const assistantNote = document.getElementById('assistant-note');
const grid = document.getElementById('recommendation-grid');
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('search-input');
const voiceButton = document.getElementById('voice-button');

const uploadModal = document.getElementById('upload-modal');
const cameraButton = document.getElementById('camera-button');
const closeModalButton = document.getElementById('close-modal');
const uploadForm = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');

const params = new URLSearchParams(window.location.search);
let currentSessionId = params.get('session');
let mediaRecorder = null;
let recorderStream = null;
let recordingChunks = [];
let recordingStopTimer = null;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle('error', isError);
}

function setVoiceButtonState(isRecording) {
  if (!voiceButton) {
    return;
  }
  voiceButton.classList.toggle('is-recording', isRecording);
  voiceButton.setAttribute(
    'aria-label',
    isRecording ? 'Stop recording and transcribe' : 'Voice to text with OpenAI'
  );
  voiceButton.title = isRecording ? 'Stop recording and transcribe' : 'Voice to text with OpenAI';
}

function preferredAudioMimeType() {
  if (typeof MediaRecorder === 'undefined') {
    return '';
  }

  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) || '';
}

function extensionFromMimeType(mimeType) {
  const normalized = (mimeType || '').toLowerCase();
  if (normalized.includes('webm')) {
    return 'webm';
  }
  if (normalized.includes('ogg')) {
    return 'ogg';
  }
  if (normalized.includes('wav')) {
    return 'wav';
  }
  if (normalized.includes('mp4') || normalized.includes('mpeg') || normalized.includes('aac')) {
    return 'm4a';
  }
  return 'webm';
}

async function transcribeAudioBlob(audioBlob) {
  if (!audioBlob || audioBlob.size === 0) {
    throw new Error('No audio captured. Please try again.');
  }

  const extension = extensionFromMimeType(audioBlob.type);
  const audioFile = new File([audioBlob], `voice-input.${extension}`, {
    type: audioBlob.type || 'audio/webm',
  });

  const formData = new FormData();
  formData.append('audio', audioFile);

  const response = await fetch('/api/transcribe', {
    method: 'POST',
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || 'Voice transcription failed.');
  }

  const text = (payload.text || '').trim();
  if (!text) {
    throw new Error('No speech was detected. Please try again.');
  }

  return text;
}

function clearRecorderTimer() {
  if (recordingStopTimer) {
    clearTimeout(recordingStopTimer);
    recordingStopTimer = null;
  }
}

function releaseRecorderStream() {
  if (recorderStream) {
    recorderStream.getTracks().forEach((track) => track.stop());
    recorderStream = null;
  }
}

async function startVoiceCapture() {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
    setStatus('Voice capture is not supported in this browser.', true);
    return;
  }

  try {
    recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordingChunks = [];

    const mimeType = preferredAudioMimeType();
    mediaRecorder = mimeType ? new MediaRecorder(recorderStream, { mimeType }) : new MediaRecorder(recorderStream);

    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) {
        recordingChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener('stop', async () => {
      clearRecorderTimer();
      releaseRecorderStream();
      setVoiceButtonState(false);

      const blobType = mediaRecorder?.mimeType || mimeType || 'audio/webm';
      const audioBlob = new Blob(recordingChunks, { type: blobType });
      mediaRecorder = null;

      setStatus('Transcribing voice with OpenAI...');
      try {
        const text = await transcribeAudioBlob(audioBlob);
        searchInput.value = text;
        searchInput.focus();
        setStatus('Voice transcription complete. Edit text if needed, then press Find Items.');
      } catch (error) {
        setStatus(error.message, true);
      }
    });

    mediaRecorder.start();
    setVoiceButtonState(true);
    setStatus('Listening... click the mic again to stop.');

    recordingStopTimer = setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }, 8000);
  } catch (error) {
    releaseRecorderStream();
    setVoiceButtonState(false);
    setStatus(error.message || 'Unable to access microphone.', true);
  }
}

function stopVoiceCapture() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
}

function toggleVoiceCapture() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    stopVoiceCapture();
    return;
  }
  startVoiceCapture();
}

function matchBlock(match) {
  if (!match) {
    return '';
  }

  const confidence = typeof match.confidence === 'number' ? ` (${Math.round(match.confidence * 100)}%)` : '';
  return `
    <div class="match-result">
      <strong>${match.verdict}${confidence}</strong>
      <p class="product-desc">${match.rationale}</p>
    </div>
  `;
}

function productCard(product) {
  const article = document.createElement('article');
  article.className = 'product-card';

  article.innerHTML = `
    <img src="${product.image_url}" alt="${product.name}" loading="lazy" />
    <div class="product-copy">
      <h4>#${product.rank} ${product.name}</h4>
      <div class="product-meta">
        <span>${product.gender}</span>
        <span>${product.article_type}</span>
        <span>${product.base_colour}</span>
        <span>${product.usage || 'Lifestyle'}</span>
      </div>
      <p class="product-desc">${product.master_category} / ${product.sub_category} | Season: ${product.season || 'All'} | Year: ${product.year || 'n/a'} | Similarity: ${product.score.toFixed(3)}</p>
      <button class="match-button" data-product-id="${product.id}">Check Your Match</button>
      ${matchBlock(product.match)}
    </div>
  `;

  const button = article.querySelector('.match-button');
  button?.addEventListener('click', () => runCheckMatch(product.id, button, article));

  return article;
}

async function loadPersonalized(sessionId) {
  grid.innerHTML = '';

  if (!sessionId) {
    assistantNote.textContent = 'Try searching or uploading something first to generate your personalized items.';
    setStatus('No recommendation session yet. Start from Home with search or image upload.', true);
    return;
  }

  setStatus('Loading your personalized recommendations...');

  try {
    const response = await fetch(`/api/personalized/${encodeURIComponent(sessionId)}`);
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || 'Could not load personalized recommendations.');
    }

    const recommendations = payload.recommendations || [];
    if (!recommendations.length) {
      assistantNote.textContent = 'Try searching or uploading something first to generate your personalized items.';
      setStatus('No recommendations available for this session.', true);
      return;
    }

    const source = payload.session?.source || 'unknown flow';
    assistantNote.textContent = `Outfit Assistant AI powered by OpenAI generated these picks from ${source}.`;

    recommendations.forEach((product) => {
      grid.appendChild(productCard(product));
    });

    setStatus('Review each recommendation and run Check Your Match for fit guidance.');
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function runCheckMatch(productId, button, card) {
  if (!currentSessionId) {
    setStatus('No session available. Run a search or upload first.', true);
    return;
  }

  button.disabled = true;
  const originalLabel = button.textContent;
  button.textContent = 'Checking...';

  try {
    const response = await fetch('/api/check-match', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: currentSessionId,
        product_id: productId,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Match check failed.');
    }

    const existing = card.querySelector('.match-result');
    if (existing) {
      existing.remove();
    }

    const confidence = typeof payload.confidence === 'number' ? ` (${Math.round(payload.confidence * 100)}%)` : '';
    const div = document.createElement('div');
    div.className = 'match-result';
    div.innerHTML = `
      <strong>${payload.verdict}${confidence}</strong>
      <p class="product-desc">${payload.rationale}</p>
    `;

    card.querySelector('.product-copy')?.appendChild(div);
    setStatus('Match check completed for selected item.');
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

async function runSearch(event) {
  event.preventDefault();
  const query = searchInput.value.trim();
  if (!query) {
    setStatus('Please enter a search query first.', true);
    return;
  }

  setStatus('Running natural-language-query-search...');

  try {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        shopper_name: 'RetailNext Shopper',
        top_k: 10,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Search failed.');
    }

    currentSessionId = payload?.session?.session_id;
    if (!currentSessionId) {
      throw new Error('Search completed but no session returned.');
    }

    window.history.replaceState({}, '', `/personalized?session=${encodeURIComponent(currentSessionId)}`);
    loadPersonalized(currentSessionId);
  } catch (error) {
    setStatus(error.message, true);
  }
}

function openUploadModal() {
  uploadModal?.showModal();
}

function closeUploadModal() {
  uploadModal?.close();
}

async function runImageMatch(event) {
  event.preventDefault();
  if (!imageInput.files?.length) {
    setStatus('Please choose an image before uploading.', true);
    return;
  }

  const formData = new FormData();
  formData.append('image', imageInput.files[0]);
  formData.append('shopper_name', 'RetailNext Shopper');
  formData.append('top_k', '10');

  closeUploadModal();
  setStatus('Running image-upload-match flow...');

  try {
    const response = await fetch('/api/image-match', {
      method: 'POST',
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Image match failed.');
    }

    currentSessionId = payload?.session?.session_id;
    if (!currentSessionId) {
      throw new Error('Image matching completed but no session returned.');
    }

    window.history.replaceState({}, '', `/personalized?session=${encodeURIComponent(currentSessionId)}`);
    loadPersonalized(currentSessionId);
  } catch (error) {
    setStatus(error.message, true);
  }
}

searchForm?.addEventListener('submit', runSearch);
cameraButton?.addEventListener('click', openUploadModal);
closeModalButton?.addEventListener('click', closeUploadModal);
uploadForm?.addEventListener('submit', runImageMatch);
voiceButton?.addEventListener('click', toggleVoiceCapture);

if (uploadModal) {
  uploadModal.addEventListener('click', (event) => {
    const rect = uploadModal.getBoundingClientRect();
    const withinDialog =
      event.clientX >= rect.left &&
      event.clientX <= rect.right &&
      event.clientY >= rect.top &&
      event.clientY <= rect.bottom;

    if (!withinDialog) {
      closeUploadModal();
    }
  });
}

loadPersonalized(currentSessionId);
