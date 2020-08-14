import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import * as React from 'react';
import { Platform, StatusBar, StyleSheet,Dimensions, View, ActivityIndicator } from 'react-native';
import Constants from 'expo-constants'

import useCachedResources from './hooks/useCachedResources';
import BottomTabNavigator from './navigation/BottomTabNavigator';
import LinkingConfiguration from './navigation/LinkingConfiguration';

const Stack = createStackNavigator();

export default function App(props) {
  const isLoadingComplete = useCachedResources();

  if (!isLoadingComplete) {
    return <ActivityIndicator style={{alignSelf:"center",flex:1,backgroundColor:"#111111",width:"100%"}} size="large" color="rgb(1,175,250)"></ActivityIndicator>;
  } else {
    return (
      <View style={styles.container}>
        <StatusBar hidden />
        {/* {Platform.OS === 'ios' && <StatusBar barStyle="dark-content" />} */}
        <NavigationContainer linking={LinkingConfiguration}>
          <Stack.Navigator>
            <Stack.Screen name="Root" component={BottomTabNavigator} />
          </Stack.Navigator>
        </NavigationContainer>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
    // paddingTop:Constants.statusBarHeight
  },
});
