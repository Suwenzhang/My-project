// 全局变量
let translateButton = null;
let translationPopup = null;
let selectedText = '';
let isTranslating = false;

// 语言映射
const languageMap = {
  'auto': '自动检测',
  'en': '英语',
  'zh': '中文',
  'ja': '日语',
  'ko': '韩语',
  'fr': '法语',
  'es': '西班牙语'
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
  initContentScript();
});

function initContentScript() {
  // 监听鼠标抬起事件（选中文本）
  document.addEventListener('mouseup', handleMouseUp);
  
  // 监听点击事件（隐藏翻译按钮和弹窗）
  document.addEventListener('click', handleClick);
  
  // 监听消息
  chrome.runtime.onMessage.addListener(handleMessage);
  
  // 创建翻译按钮和弹窗
  createTranslateButton();
  createTranslationPopup();
}

// 处理鼠标抬起事件
function handleMouseUp(event) {
  // 延迟执行，确保选择完成
  setTimeout(() => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    
    if (text && text.length <= 500) {
      selectedText = text;
      showTranslateButton(event.pageX, event.pageY);
    } else {
      hideTranslateButton();
    }
  }, 100);
}

// 处理点击事件
function handleClick(event) {
  // 如果点击的不是翻译按钮或弹窗，则隐藏它们
  if (translateButton && !translateButton.contains(event.target) && 
      translationPopup && !translationPopup.contains(event.target)) {
    hideTranslateButton();
    hideTranslationPopup();
  }
}

// 处理消息
function handleMessage(request, sender, sendResponse) {
  if (request.action === "getSelectedText") {
    sendResponse({text: selectedText});
  } else if (request.action === "translateText") {
    translateText(request.text, request.sourceLang, request.targetLang);
    sendResponse({success: true});
  }
  return true;
}

// 创建翻译按钮
function createTranslateButton() {
  translateButton = document.createElement('div');
  translateButton.id = 'quicktranslator-button';
  translateButton.className = 'quicktranslator-button';
  translateButton.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"></path>
    </svg>
    翻译
  `;
  
  translateButton.addEventListener('click', handleTranslateButtonClick);
  document.body.appendChild(translateButton);
}

// 创建翻译弹窗
function createTranslationPopup() {
  translationPopup = document.createElement('div');
  translationPopup.id = 'quicktranslator-popup';
  translationPopup.className = 'quicktranslator-popup';
  
  document.body.appendChild(translationPopup);
}

// 显示翻译按钮
function showTranslateButton(x, y) {
  if (!translateButton) return;
  
  // 确保按钮不会超出视窗
  const buttonWidth = 60;
  const buttonHeight = 30;
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;
  
  let adjustedX = x;
  let adjustedY = y;
  
  if (x + buttonWidth > windowWidth) {
    adjustedX = windowWidth - buttonWidth - 10;
  }
  
  if (y + buttonHeight > windowHeight) {
    adjustedY = windowHeight - buttonHeight - 10;
  }
  
  translateButton.style.left = `${adjustedX}px`;
  translateButton.style.top = `${adjustedY + 10}px`;
  translateButton.style.display = 'flex';
}

// 隐藏翻译按钮
function hideTranslateButton() {
  if (translateButton) {
    translateButton.style.display = 'none';
  }
}

// 处理翻译按钮点击
function handleTranslateButtonClick(event) {
  event.stopPropagation();
  
  if (!selectedText || isTranslating) return;
  
  isTranslating = true;
  
  // 获取设置
  chrome.storage.sync.get({
    targetLanguage: 'zh',
    fontSize: 'medium',
    theme: 'light'
  }, function(items) {
    translateText(selectedText, 'auto', items.targetLanguage);
  });
}

// 翻译文本
function translateText(text, sourceLang, targetLang) {
  if (!translationPopup) return;
  
  // 显示加载状态
  translationPopup.innerHTML = `
    <div class="quicktranslator-loading">
      <div class="quicktranslator-spinner"></div>
      <div>翻译中...</div>
    </div>
  `;
  
  // 计算弹窗位置
  const selection = window.getSelection();
  const range = selection.getRangeAt(0);
  const rect = range.getBoundingClientRect();
  
  showTranslationPopup(rect);
  
  // 使用Google Translate API
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sourceLang}&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`;
  
  fetch(url)
    .then(response => response.json())
    .then(data => {
      if (data && data[0] && data[0][0] && data[0][0][0]) {
        const translatedText = data[0][0][0];
        displayTranslationResult(text, translatedText, sourceLang, targetLang);
      } else {
        throw new Error('翻译结果格式错误');
      }
    })
    .catch(error => {
      console.error('翻译错误:', error);
      translationPopup.innerHTML = `
        <div class="quicktranslator-error">
          <div>翻译失败，请稍后重试</div>
        </div>
      `;
    })
    .finally(() => {
      isTranslating = false;
    });
}

// 显示翻译弹窗
function showTranslationPopup(selectionRect) {
  if (!translationPopup) return;
  
  const popupWidth = 350;
  const popupHeight = 200;
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
  
  let left = selectionRect.left + selectionRect.width / 2 - popupWidth / 2;
  let top = selectionRect.bottom + scrollTop + 10;
  
  // 确保弹窗不会超出视窗
  if (left < 10) left = 10;
  if (left + popupWidth > windowWidth - 10) left = windowWidth - popupWidth - 10;
  if (top + popupHeight > windowHeight + scrollTop - 10) {
    top = selectionRect.top + scrollTop - popupHeight - 10;
  }
  
  translationPopup.style.left = `${left}px`;
  translationPopup.style.top = `${top}px`;
  translationPopup.style.width = `${popupWidth}px`;
  translationPopup.style.display = 'block';
}

// 隐藏翻译弹窗
function hideTranslationPopup() {
  if (translationPopup) {
    translationPopup.style.display = 'none';
  }
}

// 显示翻译结果
function displayTranslationResult(originalText, translatedText, sourceLang, targetLang) {
  if (!translationPopup) return;
  
  translationPopup.innerHTML = `
    <div class="quicktranslator-header">
      <div class="quicktranslator-languages">
        <span>${languageMap[sourceLang] || '自动检测'}</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="17 1 21 5 17 9"></polyline>
          <path d="M3 11V9a4 4 0 0 1 4-4h14"></path>
          <polyline points="7 23 3 19 7 15"></polyline>
          <path d="M21 13v2a4 4 0 0 1-4 4H3"></path>
        </svg>
        <span>${languageMap[targetLang]}</span>
      </div>
      <button class="quicktranslator-close" id="quicktranslator-close">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
    <div class="quicktranslator-content">
      <div class="quicktranslator-original">
        <div class="quicktranslator-text-label">原文</div>
        <div class="quicktranslator-text">${escapeHtml(originalText)}</div>
      </div>
      <div class="quicktranslator-translated">
        <div class="quicktranslator-text-label">译文</div>
        <div class="quicktranslator-text">${escapeHtml(translatedText)}</div>
      </div>
    </div>
    <div class="quicktranslator-actions">
      <button class="quicktranslator-action-btn" id="quicktranslator-copy" title="复制译文">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
      <button class="quicktranslator-action-btn" id="quicktranslator-speak-original" title="朗读原文">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
      </button>
      <button class="quicktranslator-action-btn" id="quicktranslator-speak-translated" title="朗读译文">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
        </svg>
      </button>
    </div>
  `;
  
  // 添加事件监听器
  const closeBtn = translationPopup.querySelector('#quicktranslator-close');
  const copyBtn = translationPopup.querySelector('#quicktranslator-copy');
  const speakOriginalBtn = translationPopup.querySelector('#quicktranslator-speak-original');
  const speakTranslatedBtn = translationPopup.querySelector('#quicktranslator-speak-translated');
  
  closeBtn.addEventListener('click', hideTranslationPopup);
  copyBtn.addEventListener('click', () => copyToClipboard(translatedText));
  speakOriginalBtn.addEventListener('click', () => speakText(originalText, sourceLang));
  speakTranslatedBtn.addEventListener('click', () => speakText(translatedText, targetLang));
}

// 复制到剪贴板
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    showNotification('已复制到剪贴板');
  }).catch(() => {
    showNotification('复制失败', 'error');
  });
}

// 语音朗读
function speakText(text, lang) {
  if ('speechSynthesis' in window) {
    const langMap = {
      'en': 'en-US',
      'zh': 'zh-CN',
      'ja': 'ja-JP',
      'ko': 'ko-KR',
      'fr': 'fr-FR',
      'es': 'es-ES'
    };

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = langMap[lang] || 'en-US';
    utterance.rate = 0.9;
    
    speechSynthesis.speak(utterance);
  } else {
    showNotification('您的浏览器不支持语音朗读', 'error');
  }
}

// 显示通知
function showNotification(message, type = 'success') {
  const notification = document.createElement('div');
  notification.className = `quicktranslator-notification quicktranslator-notification-${type}`;
  notification.textContent = message;
  
  document.body.appendChild(notification);
  
  // 3秒后移除通知
  setTimeout(() => {
    if (notification.parentNode) {
      notification.parentNode.removeChild(notification);
    }
  }, 3000);
}

// HTML转义
function escapeHtml(text) {
  const map = {
    '&': '&',
    '<': '<',
    '>': '>',
    '"': '"',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

// 初始化内容脚本
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initContentScript);
} else {
  initContentScript();
}