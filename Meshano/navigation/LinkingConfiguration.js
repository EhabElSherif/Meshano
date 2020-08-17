import * as Linking from 'expo-linking';

export default {
  prefixes: [Linking.makeUrl('/')],
  config: {
    Root: {
      path: 'root',
      screens: {
        Home: 'home',
        Models: 'models',
        Preview: 'preview',
        Camera: 'camera',
        Model: 'model',
      },
    },
  },
};
