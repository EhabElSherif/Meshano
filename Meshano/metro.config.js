const { getDefaultConfig } = require('@expo/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

module.exports = {
    resolver: {
      assetExts: [...defaultConfig.resolver.assetExts, 'db','obj','mtl','dae','ply','png','jpg'],
    },
};