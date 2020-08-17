import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import * as React from 'react';

import TabBarIcon from '../components/TabBarIcon';
import HomeScreen from '../screens/HomeScreen';
import ModelsScreen from '../screens/ModelsScreen';
import CameraScreen from '../screens/CameraScreen';

const BottomTab = createBottomTabNavigator();
const INITIAL_ROUTE_NAME = 'Home';

export default function BottomTabNavigator({ navigation, route }) {
  // Set the header title on the parent stack navigator depending on the
  // currently active tab. Learn more in the documentation:
  // https://reactnavigation.org/docs/en/screen-options-resolution.html
  navigation.setOptions(
    { 
      headerStyle: {
        backgroundColor: '#111111',
        height:50,
      },
      headerTintColor: '#fff',
      headerTitle: getHeaderTitle(route) ,
      headerTitleStyle: {
        alignSelf: 'center',
        color:'rgb(1,175,250)',
        fontWeight: 'bold',
      },
    }
  );
  navigation.setOptions({headerShown:true});
  
  return (
    <BottomTab.Navigator initialRouteName={INITIAL_ROUTE_NAME}
      tabBarOptions={{
              activeTintColor: 'rgb(1,175,250)',
              inactiveTintColor: 'gray',
              showLabel:true,
              labelStyle:{
                fontWeight:"bold",
                fontSize:14
              },
              style:{
                backgroundColor:"#111111",
                height:60
              },
          }
        }
    >
      <BottomTab.Screen
        name="Camera"
        component={CameraScreen}
        options={{
          title: 'Create',
          tabBarIcon: ({ focused }) => <TabBarIcon focused={focused} name="md-add-circle" />,
        }}
      />
      <BottomTab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          title: 'Home',
          tabBarIcon: ({ focused }) => <TabBarIcon focused={focused} name="md-home" />,
        }}
      />
      <BottomTab.Screen
        name="Models"
        component={ModelsScreen}
        options={{
          title: 'Model',
          tabBarIcon: ({ focused }) => <TabBarIcon focused={focused} name="md-cube" />,
        }}
      />
    </BottomTab.Navigator>
  );
}

function getHeaderTitle(route) {
  const routeName = route.state?.routes[route.state.index]?.name ?? INITIAL_ROUTE_NAME;

  switch (routeName) {
    case 'Home':
      return 'Home';
    case 'Models':
      return 'Your Models';
    case 'Camera':
      return 'Create';
  }
}
