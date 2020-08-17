import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator, Button, Dimensions, Alert, Image, Modal } from 'react-native';
import { Camera } from 'expo-camera';
import * as Permissions from 'expo-permissions';
import { Ionicons } from '@expo/vector-icons';
import PreviewScreen from './PreviewScreen';

const MIN_NUMBER_OF_IMAGES = 20;

export default class CameraScreen extends React.Component {
	constructor(props){
		super(props)
		this.state = {
			viewGallary:false,
			isFocused:true,
			camera:null,
			hasCameraPermission: null,
			type: Camera.Constants.Type.back,
			capturedImages:Array(0),
			exif:null
		}
		this.acceptedNumber = this.acceptedNumber.bind(this);
		this.setViewGallary = this.setViewGallary.bind(this);
		this.clearImages = this.clearImages.bind(this);
		this.deleteImage = this.deleteImage.bind(this);
		
		this.props.navigation.addListener('focus', () => {
			this.setState({isFocused:true});
		});
		this.props.navigation.addListener('blur', () => {
			this.setState({isFocused:false});
		});

	}

	async componentDidMount() {
		const { status } = await Permissions.askAsync(Permissions.CAMERA);
		this.setState({ hasCameraPermission: status === 'granted' });
	}

	componentWillUnmount(){
		this.setState({capturedImages:Array(0)})
	}

	acceptedNumber = (imagesCount)=>{
		if (imagesCount >= MIN_NUMBER_OF_IMAGES){
			return {backgroundColor:"seagreen"}
		}else{
			return {backgroundColor:"crimson"}
		}
	}

	setViewGallary = (viewGallary)=>{
		this.setState({viewGallary:viewGallary})
	}

	clearImages = ()=>{
		this.setState({capturedImages:Array(0)})
	}

	deleteImage = (id)=>{
		console.log(this.state.capturedImages)
		let capturedImages = this.state.capturedImages
		const index = capturedImages.indexOf(id)
		if (index!=-1){
			capturedImages.splice(index,1)
		}
		this.setState({capturedImages:capturedImages})
		console.log(this.state.capturedImages)
	}

	render(){
		let capturedImages = this.state.capturedImages;
		if (this.state.hasCameraPermission === null) {
			return <View style={styles.cameraContainer}></View>;
		} else if (this.state.hasCameraPermission === false) {
			return <Text>No access to camera</Text>;
		} else {
			return (
				<View style={styles.cameraContainer}>
					{this.state.isFocused && (
						<View style={{position:'relative',flex:1}}>
							<Camera style={styles.cameraComponent}
								type={this.type}
								autoFocus={Camera.Constants.AutoFocus.on}
								ref={ref => { this.state.camera = ref; }}></Camera>
							<TouchableOpacity style={styles.reverseButton}
								activeOpacity={0.8}
								onPress={() => {
									this.setState({
										type: this.state.camera.type === Camera.Constants.Type.back ? 
										Camera.Constants.Type.front
										:
										Camera.Constants.Type.back
									});
								}}>
								<Ionicons style={{fontSize:30,color:"white"}} name="md-reverse-camera" ></Ionicons>
							</TouchableOpacity>
							<TouchableOpacity style={styles.previewButton} activeOpacity={1}
								onPress = {()=>{
									this.setViewGallary(true)
								}}
							>
								<Image style={styles.previewButtonImage} resizeMode="stretch" resizeMethod="resize" source={{uri:this.state.capturedImages[this.state.capturedImages.length-1]}}></Image>
								<Text style={[styles.previewButtonText, this.acceptedNumber(capturedImages.length)]}>{capturedImages.length}/{MIN_NUMBER_OF_IMAGES}</Text>
							</TouchableOpacity>
							<TouchableOpacity style={styles.captureButton} activeOpacity={0.6}
								onPress={() => {
									this.state.camera.takePictureAsync({
										'quality':1,
										'skipProcessing':true,
										'exif':false
									}).then(({ uri, width, height, exif, base64 })=>{
										capturedImages.push(uri);
										this.setState(capturedImages)
									})
								}}
							>
							<Ionicons style={{fontSize:30,color:"rgb(1,175,250)"}} name="md-camera"></Ionicons>
							</TouchableOpacity>
						</View>
					)}
					<Modal
						visible={this.state.viewGallary}
						animationType="slide"
						transparent={true}
						onRequestClose = {()=>{
							this.setState({viewGallary:false})
						}}
					>
						<PreviewScreen MIN_NUMBER_OF_IMAGES={MIN_NUMBER_OF_IMAGES} capturedImages={this.state.capturedImages} exif={this.state.exif} clearImages={this.clearImages} deleteImage={this.deleteImage} setViewGallary={this.setViewGallary}></PreviewScreen>
					</Modal>
				</View>
			);
		}  
	}
}

const styles = StyleSheet.create({
	cameraContainer:{
		flex: 1 ,
		position:'relative',
		backgroundColor:"#111111"

	},
	cameraComponent:{
		width:'100%',
		height:'100%',
		backgroundColor:"#111111"
	},
	reverseButton:{
		position:"absolute",
		top:25,
		left:Dimensions.get('window').width - 75,
		width:50,
		height:50,
		borderRadius:100,
		justifyContent:'center',
		alignItems:'center',
		backgroundColor:'rgba(0,0,0,0.5)'
	},
	captureButton:{
		position:'absolute',
		top:"85%",
		justifyContent:"center",
		alignItems:"center",
		alignSelf:'center',
		width:60,
		height:60,
		borderRadius:100,
		borderWidth:2,
		borderColor:'rgb(255,255,255)',
		backgroundColor:'rgba(0,0,0,0.5)'
	},
	previewButton:{
		position:"absolute",
		top:"85%",
		left:"5%",
		width:70,
		height:70,
		justifyContent:"center",
		alignItems:"center",
		backgroundColor:"#00000099",
		borderWidth:StyleSheet.hairlineWidth,
		borderColor:"#FFFFFF55",
		borderRadius:15,
		overflow:"hidden",
	},
	previewButtonImage:{
		width:"100%",
		height:"100%",
		overflow:"hidden"
	},
	previewButtonText:{
		fontSize:15,
		marginTop:-20,
		color:"#bdbdbd",
		width:"100%",
		margin:0,
		textAlign:"center",
	}
})