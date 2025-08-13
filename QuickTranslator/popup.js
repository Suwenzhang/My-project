document.addEventListener('DOMContentLoaded', function() {
  // 获取DOM元素
  const sourceLangSelect = document.getElementById('source-lang');
  const targetLangSelect = document.getElementById('target-lang');
  const swapLangsBtn = document.getElementById('swap-langs');
  const inputText = document.getElementById('input-text');
  const charCount = document.getElementById('char-count');
  const translateBtn = document.getElementById('translate-btn');
  const translationResult = document.getElementById('translation-result');
  const fontDecreaseBtn = document.getElementById('font-decrease');
  const fontIncreaseBtn = document.getElementById('font-increase');
  const copyResultBtn = document.getElementById('copy-result');
  const speakSourceBtn = document.getElementById('speak-source');
  const speakTargetBtn = document.getElementById('speak-target');
  const settingsBtn = document.getElementById('settings-btn');
  const status = document.getElementById('status');

  // 字体大小级别
  let fontSizeLevel = 1; // 0: 小, 1: 中, 2: 大

  // 从存储中加载设置
  chrome.storage.sync.get({
    targetLanguage: 'zh',
    fontSize: 'medium',
    theme: 'light'
  }, function(items) {
    targetLangSelect.value = items.targetLanguage;
    
    // 设置字体大小
    switch(items.fontSize) {
      case 'small':
        fontSizeLevel = 0;
        break;
      case 'medium':
        fontSizeLevel = 1;
        break;
      case 'large':
        fontSizeLevel = 2;
        break;
    }
    updateFontSize();
  });

  // 字符计数
  inputText.addEventListener('input', function() {
    const count = this.value.length;
    charCount.textContent = count;
    
    if (count > 500) {
      this.value = this.value.substring(0, 500);
      charCount.textContent = 500;
    }
  });

  // 交换语言
  swapLangsBtn.addEventListener('click', function() {
    if (sourceLangSelect.value !== 'auto') {
      const temp = sourceLangSelect.value;
      sourceLangSelect.value = targetLangSelect.value;
      targetLangSelect.value = temp;
    }
  });

  // 翻译按钮点击事件
  translateBtn.addEventListener('click', function() {
    const text = inputText.value.trim();
    if (!text) {
      updateStatus('请输入要翻译的文本', 'error');
      return;
    }

    translateText(text, sourceLangSelect.value, targetLangSelect.value);
  });

  // 字体大小调节
  fontDecreaseBtn.addEventListener('click', function() {
    if (fontSizeLevel > 0) {
      fontSizeLevel--;
      updateFontSize();
    }
  });

  fontIncreaseBtn.addEventListener('click', function() {
    if (fontSizeLevel < 2) {
      fontSizeLevel++;
      updateFontSize();
    }
  });

  // 复制翻译结果
  copyResultBtn.addEventListener('click', function() {
    const resultText = translationResult.textContent;
    if (resultText && resultText !== '翻译结果将显示在这里') {
      navigator.clipboard.writeText(resultText).then(function() {
        updateStatus('已复制到剪贴板', 'success');
      }).catch(function() {
        updateStatus('复制失败', 'error');
      });
    }
  });

  // 朗读原文
  speakSourceBtn.addEventListener('click', function() {
    const text = inputText.value.trim();
    if (text) {
      speakText(text, sourceLangSelect.value);
    }
  });

  // 朗读译文
  speakTargetBtn.addEventListener('click', function() {
    const resultText = translationResult.textContent;
    if (resultText && resultText !== '翻译结果将显示在这里') {
      speakText(resultText, targetLangSelect.value);
    }
  });

  // 设置按钮
  settingsBtn.addEventListener('click', function() {
    chrome.runtime.openOptionsPage();
  });

  // 翻译文本函数
  function translateText(text, sourceLang, targetLang) {
    updateStatus('翻译中...', 'loading');
    translateBtn.disabled = true;

    // 使用Google Translate API
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sourceLang}&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`;

    fetch(url)
      .then(response => response.json())
      .then(data => {
        if (data && data[0] && data[0][0] && data[0][0][0]) {
          const translatedText = data[0][0][0];
          displayTranslation(text, translatedText);
          updateStatus('翻译完成', 'success');
        } else {
          throw new Error('翻译结果格式错误');
        }
      })
      .catch(error => {
        console.error('翻译错误:', error);
        updateStatus('翻译失败，请稍后重试', 'error');
      })
      .finally(() => {
        translateBtn.disabled = false;
      });
  }

  // 显示翻译结果
  function displayTranslation(originalText, translatedText) {
    translationResult.innerHTML = `
      <div class="translation-content">
        <div class="source-text">
          <div class="text-label">原文</div>
          <div class="text-content">${escapeHtml(originalText)}</div>
        </div>
        <div class="target-text">
          <div class="text-label">译文</div>
          <div class="text-content">${escapeHtml(translatedText)}</div>
        </div>
      </div>
    `;
  }

  // 更新字体大小
  function updateFontSize() {
    const sizes = ['small', 'medium', 'large'];
    translationResult.className = `translation-result font-${sizes[fontSizeLevel]}`;
  }

  // 更新状态
  function updateStatus(message, type = 'info') {
    status.textContent = message;
    status.className = `status ${type}`;
    
    // 如果是成功或错误消息，3秒后恢复默认状态
    if (type === 'success' || type === 'error') {
      setTimeout(() => {
        status.textContent = '就绪';
        status.className = 'status';
      }, 3000);
    }
  }

  // 语音朗读
  function speakText(text, lang) {
    if ('speechSynthesis' in window) {
      // 映射语言代码到语音合成支持的语言代码
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
      updateStatus('正在朗读...', 'info');
    } else {
      updateStatus('您的浏览器不支持语音朗读', 'error');
    }
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

  // 从内容脚本获取选中的文本
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.tabs.sendMessage(tabs[0].id, {action: "getSelectedText"}, function(response) {
      if (response && response.text) {
        inputText.value = response.text;
        charCount.textContent = response.text.length;
      }
    });
  });
});