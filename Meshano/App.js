import React, { Component } from 'react';
import { StyleSheet, View, AppRegistry,StatusBar, Alert } from 'react-native';
import Constants from 'expo-constants';
import TabBar from './components/TabBar/TabBar';
import Home from './components/Home/Home';
import CameraScreen from './components/Camera/Camera';
import AllModels from './components/AllModels/AllModels';
import 'react-native-gesture-handler';
import {NavigationContainer} from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { TouchableOpacity } from 'react-native-gesture-handler';

const Tab = createBottomTabNavigator();

export default class App extends Component {
	render() {
		return (
			<NavigationContainer style={styles.container}>
				<Tab.Navigator initialRouteName="Home"
					screenOptions={
						({ route }) => ({
							tabBarIcon: ({ focused, color, size }) => {
								let iconName;
					
								if (route.name === 'Home') {
									iconName = 'md-home';
								} else if (route.name === 'View') {
									iconName = 'md-cube';
								} else if (route.name === 'Create'){
									iconName = 'md-add';
								}
								return <Ionicons name={iconName} size={size} color={color} />;
							},
						})
					}
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
					}}
					backBehavior="history"
				>
					<Tab.Screen name="Create" component={CameraScreen}></Tab.Screen>
					<Tab.Screen name="Home" component={Home}/>
					<Tab.Screen name="View" component={AllModels}/>
				</Tab.Navigator>
				{/* <TabBar></TabBar> */}
			</NavigationContainer>
		)
	}
}

const styles = StyleSheet.create({
	// dark = #111111
	// gray = #dfdfdf
	// white = #ffffff
	// primary blue = #01afd1 rgb(1,175,250)
	container: {
		flex: 1,
		backgroundColor:"#111111",
		marginTop:Constants.statusBarHeight,
		// alignItems: 'center',
		// justifyContent: 'center',
	},
});
AppRegistry.registerComponent('App',()=>App)