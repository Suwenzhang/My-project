document.addEventListener('DOMContentLoaded', function() {
  // 获取DOM元素
  const targetLanguage = document.getElementById('target-language');
  const autoDetect = document.getElementById('auto-detect');
  const showTranslateButton = document.getElementById('show-translate-button');
  const fontSizeRadios = document.querySelectorAll('input[name="font-size"]');
  const themeRadios = document.querySelectorAll('input[name="theme"]');
  const enableShortcut = document.getElementById('enable-shortcut');
  const currentShortcut = document.getElementById('current-shortcut');
  const changeShortcutBtn = document.getElementById('change-shortcut');
  const translationService = document.getElementById('translation-service');
  const maxTextLength = document.getElementById('max-text-length');
  const enableTts = document.getElementById('enable-tts');
  const saveSettingsBtn = document.getElementById('save-settings');
  const resetSettingsBtn = document.getElementById('reset-settings');
  const statusMessage = document.getElementById('status-message');
  const feedbackLink = document.getElementById('feedback-link');
  const aboutLink = document.getElementById('about-link');
  
  // 快捷键对话框元素
  const shortcutDialog = document.getElementById('shortcut-dialog');
  const closeShortcutDialog = document.getElementById('close-shortcut-dialog');
  const shortcutInput = document.getElementById('shortcut-input');
  const saveShortcutBtn = document.getElementById('save-shortcut');
  const cancelShortcutBtn = document.getElementById('cancel-shortcut');
  
  // 关于对话框元素
  const aboutDialog = document.getElementById('about-dialog');
  const closeAboutDialog = document.getElementById('close-about-dialog');
  const closeAboutBtn = document.getElementById('close-about-btn');
  
  // 默认设置
  const defaultSettings = {
    targetLanguage: 'zh',
    autoDetect: true,
    showTranslateButton: true,
    fontSize: 'medium',
    theme: 'light',
    enableShortcut: true,
    shortcut: 'Ctrl+Shift+T',
    translationService: 'google',
    maxTextLength: 500,
    enableTts: true
  };
  
  // 当前设置
  let currentSettings = {...defaultSettings};
  let newShortcut = '';
  
  // 加载设置
  loadSettings();
  
  // 事件监听器
  saveSettingsBtn.addEventListener('click', saveSettings);
  resetSettingsBtn.addEventListener('click', resetSettings);
  changeShortcutBtn.addEventListener('click', showShortcutDialog);
  closeShortcutDialog.addEventListener('click', hideShortcutDialog);
  cancelShortcutBtn.addEventListener('click', hideShortcutDialog);
  saveShortcutBtn.addEventListener('click', saveShortcut);
  feedbackLink.addEventListener('click', handleFeedback);
  aboutLink.addEventListener('click', showAboutDialog);
  closeAboutDialog.addEventListener('click', hideAboutDialog);
  closeAboutBtn.addEventListener('click', hideAboutDialog);
  
  // 快捷键输入监听
  shortcutInput.addEventListener('keydown', handleShortcutInput);
  shortcutInput.addEventListener('click', function() {
    this.textContent = '按下快捷键组合...';
    this.focus();
  });
  
  // 点击对话框外部关闭对话框
  shortcutDialog.addEventListener('click', function(e) {
    if (e.target === shortcutDialog) {
      hideShortcutDialog();
    }
  });
  
  aboutDialog.addEventListener('click', function(e) {
    if (e.target === aboutDialog) {
      hideAboutDialog();
    }
  });
  
  // 加载设置
  function loadSettings() {
    chrome.storage.sync.get(defaultSettings, function(items) {
      currentSettings = {...items};
      
      // 应用设置到UI
      targetLanguage.value = items.targetLanguage;
      autoDetect.checked = items.autoDetect;
      showTranslateButton.checked = items.showTranslateButton;
      
      // 设置字体大小
      fontSizeRadios.forEach(radio => {
        radio.checked = radio.value === items.fontSize;
      });
      
      // 设置主题
      themeRadios.forEach(radio => {
        radio.checked = radio.value === items.theme;
      });
      
      // 设置其他选项
      enableShortcut.checked = items.enableShortcut;
      currentShortcut.textContent = items.shortcut;
      translationService.value = items.translationService;
      maxTextLength.value = items.maxTextLength;
      enableTts.checked = items.enableTts;
      
      // 应用主题
      applyTheme(items.theme);
    });
  }
  
  // 保存设置
  function saveSettings() {
    // 收集设置
    const settings = {
      targetLanguage: targetLanguage.value,
      autoDetect: autoDetect.checked,
      showTranslateButton: showTranslateButton.checked,
      fontSize: document.querySelector('input[name="font-size"]:checked').value,
      theme: document.querySelector('input[name="theme"]:checked').value,
      enableShortcut: enableShortcut.checked,
      shortcut: currentSettings.shortcut,
      translationService: translationService.value,
      maxTextLength: parseInt(maxTextLength.value),
      enableTts: enableTts.checked
    };
    
    // 验证设置
    if (settings.maxTextLength < 100 || settings.maxTextLength > 1000) {
      showStatus('最大文本长度必须在100到1000之间', 'error');
      return;
    }
    
    // 保存设置
    chrome.storage.sync.set(settings, function() {
      currentSettings = {...settings};
      showStatus('设置已保存', 'success');
      
      // 应用主题
      applyTheme(settings.theme);
      
      // 通知背景脚本设置已更改
      chrome.runtime.sendMessage({
        action: 'updateSettings',
        settings: settings
      });
    });
  }
  
  // 重置设置
  function resetSettings() {
    if (confirm('确定要恢复默认设置吗？')) {
      chrome.storage.sync.set(defaultSettings, function() {
        currentSettings = {...defaultSettings};
        loadSettings();
        showStatus('设置已恢复默认', 'success');
        
        // 应用主题
        applyTheme(defaultSettings.theme);
        
        // 通知背景脚本设置已更改
        chrome.runtime.sendMessage({
          action: 'updateSettings',
          settings: defaultSettings
        });
      });
    }
  }
  
  // 显示快捷键对话框
  function showShortcutDialog() {
    shortcutDialog.style.display = 'flex';
    shortcutInput.textContent = '按下快捷键组合...';
    newShortcut = '';
  }
  
  // 隐藏快捷键对话框
  function hideShortcutDialog() {
    shortcutDialog.style.display = 'none';
  }
  
  // 保存快捷键
  function saveShortcut() {
    if (newShortcut) {
      currentSettings.shortcut = newShortcut;
      currentShortcut.textContent = newShortcut;
      showStatus('快捷键已更新', 'success');
    }
    hideShortcutDialog();
  }
  
  // 处理快捷键输入
  function handleShortcutInput(e) {
    e.preventDefault();
    
    const key = e.key;
    const modifiers = [];
    
    if (e.ctrlKey || e.metaKey) modifiers.push('Ctrl');
    if (e.shiftKey) modifiers.push('Shift');
    if (e.altKey) modifiers.push('Alt');
    
    // 只允许字母、数字和功能键作为主键
    if ((key.length === 1 && /[a-zA-Z0-9]/.test(key)) || 
        (key.startsWith('F') && key.length > 1 && key.length <= 3)) {
      const shortcut = [...modifiers, key.toUpperCase()].join('+');
      
      // 验证快捷键是否有效
      if (isValidShortcut(shortcut)) {
        newShortcut = shortcut;
        shortcutInput.textContent = shortcut;
      }
    }
  }
  
  // 验证快捷键是否有效
  function isValidShortcut(shortcut) {
    // 基本验证规则
    const validShortcuts = [
      /^Ctrl\+[A-Z0-9]$/,
      /^Ctrl\+Shift\+[A-Z0-9]$/,
      /^Ctrl\+Alt\+[A-Z0-9]$/,
      /^Ctrl\+Shift\+Alt\+[A-Z0-9]$/,
      /^F[1-9]$|^F1[0-2]$/
    ];
    
    return validShortcuts.some(pattern => pattern.test(shortcut));
  }
  
  // 显示关于对话框
  function showAboutDialog(e) {
    e.preventDefault();
    aboutDialog.style.display = 'flex';
  }
  
  // 隐藏关于对话框
  function hideAboutDialog() {
    aboutDialog.style.display = 'none';
  }
  
  // 处理反馈
  function handleFeedback(e) {
    e.preventDefault();
    // 这里可以打开反馈页面或发送邮件
    window.open('mailto:support@quicktranslator.com?subject=QuickTranslator反馈', '_blank');
  }
  
  // 显示状态消息
  function showStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message status-${type}`;
    
    // 3秒后清除状态消息
    setTimeout(() => {
      statusMessage.textContent = '';
      statusMessage.className = 'status-message';
    }, 3000);
  }
  
  // 应用主题
  function applyTheme(theme) {
    if (theme === 'dark') {
      document.body.classList.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
    }
  }
  
  // 监听设置变化
  chrome.storage.onChanged.addListener(function(changes, namespace) {
    if (namespace === 'sync') {
      for (let key in changes) {
        if (key === 'theme') {
          applyTheme(changes[key].newValue);
        }
      }
    }
  });
});