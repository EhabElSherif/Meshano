import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Camera } from 'expo-camera';
import * as Permissions from 'expo-permissions';
import { useFocusEffect } from '@react-navigation/native';
import { useIsFocused } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';

export default function CameraScreen({navigation}) {
	const [camera, setCamera] = useState({
		hasCameraPermission: null,
		type: Camera.Constants.Type.back,
	});
	const isFocused = useIsFocused();

	useEffect(() => {
		(async () => {
		  	const { status } = await Camera.requestPermissionsAsync();
		  	setCamera(prevState => ({ ...prevState, hasCameraPermission: status === 'granted'}));
		})();
	  }, []);
	  
	// useEffect(async () => {
	// 	const { status } = await Permissions.askAsync(Permissions.CAMERA);
	// 	setCamera(prevState => ({ ...prevState, hasCameraPermission: status === 'granted'}));

	// }, []);
	
	if (camera.hasCameraPermission === null) {
		return <View />;
	} else if (camera.hasCameraPermission === false) {
		return <Text>No access to camera</Text>;
	} else {
		return (
			<View style={{ flex: 1 ,backgroundColor:"#111111"}}>
				{isFocused ? 
					<Camera style={{ flex: 1 ,backgroundColor:"#111111"}} type={camera.type}>
						<View style={{flex: 1,backgroundColor: 'transparent',flexDirection: 'row'}}>
							<TouchableOpacity style={{flex: 1,alignSelf: 'flex-start',alignItems:"flex-end",marginTop: 50,marginRight:25}}
								onPress={() => {
									// setCamera({
									// 	type: camera.type === Camera.Constants.Type.back ? 
									// 		Camera.Constants.Type.front
									// 		:
									// 		Camera.Constants.Type.back
									// });
									fetch("http://192.168.43.36:5000/")
									.then(response => {
										console.log(response);
										response.json()
									})
									.then(json => {
									  	console.log(json);
									})
									.catch((error) => {
									  	console.error(error);
									});
									}}>
								<Ionicons style={{fontSize:30,color:"white"}} name="md-reverse-camera" ></Ionicons>
							</TouchableOpacity>
						</View>
					</Camera>
				:
				<ActivityIndicator style={{alignSelf:"center",flex:1,backgroundColor:"#111111",width:"100%"}} size="large" color="rgb(1,175,250)"></ActivityIndicator>
				}
			</View>
		);
	}  
}