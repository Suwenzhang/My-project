// 后台脚本
let contextMenuId = null;

// 默认设置
const defaultSettings = {
  targetLanguage: 'zh',
  fontSize: 'medium',
  theme: 'light',
  autoDetect: true,
  showTranslateButton: true,
  enableShortcut: true
};

// 初始化
chrome.runtime.onInstalled.addListener(function(details) {
  // 初始化设置
  initializeSettings();
  
  // 创建右键菜单
  createContextMenus();
  
  // 显示安装通知
  if (details.reason === 'install') {
    showNotification('QuickTranslator 已安装成功！', 'success');
  } else if (details.reason === 'update') {
    showNotification('QuickTranslator 已更新到新版本！', 'info');
  }
});

// 初始化设置
function initializeSettings() {
  chrome.storage.sync.get(defaultSettings, function(items) {
    chrome.storage.sync.set(items, function() {
      console.log('设置已初始化');
    });
  });
}

// 创建右键菜单
function createContextMenus() {
  // 移除已存在的菜单
  if (contextMenuId) {
    chrome.contextMenus.remove(contextMenuId);
  }
  
  // 创建翻译选中文本的右键菜单
  contextMenuId = chrome.contextMenus.create({
    id: 'quicktranslator-translate',
    title: '翻译 "%s"',
    contexts: ['selection']
  });
  
  // 创建设置菜单
  chrome.contextMenus.create({
    id: 'quicktranslator-settings',
    title: 'QuickTranslator 设置',
    contexts: ['all']
  });
}

// 监听右键菜单点击
chrome.contextMenus.onClicked.addListener(function(info, tab) {
  if (info.menuItemId === 'quicktranslator-translate' && info.selectionText) {
    // 发送消息到内容脚本进行翻译
    chrome.tabs.sendMessage(tab.id, {
      action: 'translateFromContextMenu',
      text: info.selectionText
    });
  } else if (info.menuItemId === 'quicktranslator-settings') {
    // 打开设置页面
    chrome.runtime.openOptionsPage();
  }
});

// 监听快捷键命令
chrome.commands.onCommand.addListener(function(command) {
  if (command === 'translate-selection') {
    // 获取当前活动标签页
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      if (tabs[0]) {
        // 发送消息到内容脚本获取选中文本并翻译
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'translateFromShortcut'
        });
      }
    });
  }
});

// 监听来自内容脚本和弹出页面的消息
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  switch (request.action) {
    case 'getSettings':
      // 获取设置
      chrome.storage.sync.get(defaultSettings, function(settings) {
        sendResponse({settings: settings});
      });
      return true;
      
    case 'updateSettings':
      // 更新设置
      chrome.storage.sync.set(request.settings, function() {
        sendResponse({success: true});
      });
      return true;
      
    case 'translateText':
      // 翻译文本
      translateText(request.text, request.sourceLang, request.targetLang)
        .then(result => sendResponse({success: true, result: result}))
        .catch(error => sendResponse({success: false, error: error.message}));
      return true;
      
    case 'showNotification':
      // 显示通知
      showNotification(request.message, request.type);
      sendResponse({success: true});
      return true;
  }
});

// 翻译文本函数
async function translateText(text, sourceLang, targetLang) {
  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sourceLang}&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`;
    
    const response = await fetch(url);
    const data = await response.json();
    
    if (data && data[0] && data[0][0] && data[0][0][0]) {
      return {
        originalText: text,
        translatedText: data[0][0][0],
        sourceLang: sourceLang,
        targetLang: targetLang
      };
    } else {
      throw new Error('翻译结果格式错误');
    }
  } catch (error) {
    console.error('翻译错误:', error);
    throw error;
  }
}

// 显示通知
function showNotification(message, type = 'info') {
  // 创建通知选项
  const options = {
    type: type === 'error' ? 'basic' : 'basic',
    iconUrl: 'icons/icon48.png',
    title: 'QuickTranslator',
    message: message
  };
  
  // 显示通知
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icons/icon48.png',
    title: 'QuickTranslator',
    message: message
  });
}

// 监听通知点击
chrome.notifications.onClicked.addListener(function(notificationId) {
  // 点击通知时打开设置页面
  chrome.runtime.openOptionsPage();
  chrome.notifications.clear(notificationId);
});

// 监听存储变化
chrome.storage.onChanged.addListener(function(changes, namespace) {
  // 如果设置发生变化，可以在这里执行相应的操作
  for (let key in changes) {
    if (key === 'theme') {
      // 主题变化时，可以通知所有标签页更新主题
      chrome.tabs.query({}, function(tabs) {
        for (let i = 0; i < tabs.length; i++) {
          chrome.tabs.sendMessage(tabs[i].id, {
            action: 'updateTheme',
            theme: changes[key].newValue
          });
        }
      });
    }
  }
});

// 处理插件图标点击
chrome.action.onClicked.addListener(function(tab) {
  // 打开弹出页面
  chrome.action.openPopup();
});

// 监听标签页更新
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
  // 如果标签页加载完成，检查是否需要注入内容脚本
  if (changeInfo.status === 'complete' && tab.url && tab.url.startsWith('http')) {
    // 可以在这里执行一些初始化操作
  }
});

// 监听标签页激活
chrome.tabs.onActivated.addListener(function(activeInfo) {
  // 当用户切换到新标签页时，可以执行一些操作
  chrome.tabs.get(activeInfo.tabId, function(tab) {
    // 可以在这里执行一些操作
  });
});

// 导出函数供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    translateText: translateText,
    showNotification: showNotification
  };
}