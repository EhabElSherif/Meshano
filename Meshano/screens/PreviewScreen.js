import { Ionicons } from '@expo/vector-icons';
import * as React from 'react';
import { StyleSheet, Dimensions, Image, Text, View, TouchableOpacity, Alert, Modal, ActivityIndicator } from 'react-native';
import { ScrollView } from 'react-native-gesture-handler';
import { Video } from 'expo-av';

import AsyncStorage from '@react-native-community/async-storage';

const windowWidth = Dimensions.get('window').width;
var IMAGES_PER_ROW = 4;
const SERVER_IP = 'http//192.168.1.3';

export default class PreviewScreen extends React.Component {
	constructor(props){
		super(props)
		this.state = {
			processModel:false
		}
		this.acceptedNumber = this.acceptedNumber.bind(this)
		this.calculatedSize = this.calculatedSize.bind(this)
		this.scale = this.scale.bind(this) 
		this.renderRow = this.renderRow.bind(this) 
		this.renderRows = this.renderRows.bind(this)
		this.clearImages = this.clearImages.bind(this)
		this.handleSubmit = this.handleSubmit.bind(this)
	}

	clearImages = ()=>{
		Alert.alert(  
            'Clear Gallery',
            'Are you sure to clear all captured images?',  
            [  
                {  
                    text: 'Cancel',
                    style: 'cancel',
                },  
                {text: 'OK', onPress: () => this.props.clearImages()},  
            ],  
            {cancelable: false}  
        )  
	}

	removeImage = (image)=>{
		Alert.alert(  
			'Remove Image',  
			'Are you sure to delete this image ?',  
			[  
				{  
					text: 'Cancel',
					style: 'cancel',  
				},  
				{text: 'OK', onPress: () => this.props.deleteImage(image)},  
			],  
			{cancelable: false}  
		)
	}

	handleSubmit = async ()=>{
		if (this.props.capturedImages.length < this.props.MIN_NUMBER_OF_IMAGES){
			Alert.alert(
				"Insufficient Number of Photos",
				"Make sure to take at leaset "+this.props.MIN_NUMBER_OF_IMAGES+" photos and try again",
				[
					{text: 'OK'},  
				],  
				{cancelable: false}  
			)
			return;
		}else{
			console.log("Submit To Server")
			this.setState({processModel:true})
			const data = new FormData();
			const codec = 'jpg'
			const type = `image/${codec}`;

			this.props.capturedImages.map((uri,idx)=>{
				data.append("file[]",{
					name: "IMG_"+idx+"",
					type,
					uri
				});
			})
			console.log("SEND IMAGES")
			try {
				await fetch('http://192.168.1.3:5000/sendimages', 
				{
					method: "post",
					headers:{
						Accept: 'application/json',
						'Content-Type': 'multipart/form-data'
					},
					body: data
				}).then(response=>
					response.json()
				).then(responseJSON=>
					console.log(responseJSON['message'])
				).catch(error => {
					console.log(JSON.stringify(error,['name','type','message']));
				});
			} catch (e) {
				console.error(e);
			}
		}
	}
	acceptedNumber(imagesCount){
		if (imagesCount >= this.props.MIN_NUMBER_OF_IMAGES){
			return {backgroundColor:"seagreen",color:"#9d9d9d"}
		}else{
			return {backgroundColor:"crimson",color:"#cccccc"}
		}
	}

	calculatedSize(){
		var size = windowWidth / IMAGES_PER_ROW
		return {width: size, height: size}
	}
	
	scale(){
		return {transform:[{scale:0.9}]}
	}
	
	renderRow(images){
		return images.map((image,i)=>{
			return (
				<View key={i}>
					<Image source={{uri:image}} style={[styles.imageElement, this.calculatedSize(), this.scale()]}></Image>
					<Ionicons style={styles.removeImage} name="md-close"
						onPress={()=>this.removeImage(image)}
					></Ionicons>
				</View>
			)
		});
	}
	renderRows(images) {
		let numberOfRows = Math.floor(images.length / IMAGES_PER_ROW);
		let rows = []
		for (let i = 0; i < numberOfRows; i++) {
			let modelsForRow = images.slice(i*IMAGES_PER_ROW,IMAGES_PER_ROW*(i+1))
			rows.push(
				<View style={styles.row} key={i}>
					{this.renderRow(modelsForRow)}
				</View>
			)
		}
	
		if(images.length%IMAGES_PER_ROW != 0){
			let modelsForRow = images.slice(numberOfRows*IMAGES_PER_ROW,images.length)
			rows.push(
				<View style={styles.row} key={numberOfRows*IMAGES_PER_ROW}>
					{this.renderRow(modelsForRow)}
				</View>
			)
		}
		return rows;
	}

  	render(){
		let capturedImages = this.props.capturedImages;
		return(
			<View style={styles.container}>
				<Text style={styles.previewTitle}>Preview</Text>
				<ScrollView style={{transform:[{scaleX:0.95}]}}>
					<TouchableOpacity style={[styles.clearButton]} activeOpacity={0.7}
						onPress={this.clearImages}
					>
						<Ionicons style={{fontSize:30,color:"#111111"}} name="md-close"></Ionicons>
					</TouchableOpacity>
					{this.renderRows(capturedImages)}
				</ScrollView>
				<TouchableOpacity style={[styles.button,styles.backButton]} activeOpacity={0.8}
					onPress= {()=>{
						this.props.setViewGallary(false);
					}}
				>
					<Ionicons style={styles.submitButtonIcon} name="md-arrow-back"></Ionicons>
				</TouchableOpacity>
				<Text style={[styles.imagesCount, this.acceptedNumber(capturedImages.length)]}>{capturedImages.length}/{this.props.MIN_NUMBER_OF_IMAGES}</Text>	
				<TouchableOpacity style={[styles.button,styles.submitButton]} activeOpacity={0.8}
					onPress={this.handleSubmit}
				>
					<Ionicons style={styles.submitButtonIcon} name="md-checkmark"></Ionicons>
				</TouchableOpacity>
				<Modal
					visible={this.state.processModel}
					animationType="slide"
					transparent={true}
				>	
					<View style={styles.container}>
						<Text style={{
							marginTop:70,
							marginBottom:30,
							color:"rgb(1,175,250)",
							fontSize:25,
							fontWeight:"900",
							textAlign:"center",
							alignSelf:"center"
						}}>
							Your Model is being processed
						</Text>
						<ActivityIndicator style={{
							marginVertical:20,
							marginBottom:50,
							fontWeight:"900",
							textAlign:"center",
							alignSelf:"center"
						}}
							color={"rgb(1,175,250)"}
							size={30}
						>	
						</ActivityIndicator>
						<Video
							source={require('../assets/video/loading.mp4')}
							rate={1.0}
							volume={1.0}
							isMuted={false}
							resizeMode="stretch"
							shouldPlay={this.state.processModel}
							isLooping={true}
							style={{ width: "80%", height: "50%", alignSelf:"center",marginBottom:10}}
						/>
						<TouchableOpacity style={[styles.backButton,{alignSelf:"center",left:0}]} activeOpacity={0.7}
							onPress={()=>{this.setState({processModel:false})}}
						>
							<Ionicons style={{fontSize:40,color:"crimson"}} name="md-close-circle-outline"></Ionicons>
						</TouchableOpacity>

					</View>
				</Modal>
			</View>
		);
	}
}

const styles = StyleSheet.create({
	container: {
		flex: 1,
		flexDirection:'column',
		backgroundColor: '#111111',
	},
	previewTitle:{
		marginVertical:10,
		height:50,
		fontSize:20,
        alignSelf: 'center',
        color:'rgb(1,175,250)',
        fontWeight: 'bold',
        backgroundColor: '#111111',
	},
	row:{
		flexDirection:'row',
	},
	clearButton:{
		width:50,
		height:50,
		alignItems:"center",
		justifyContent:"center",
		borderRadius:100,
		backgroundColor:"crimson",
		marginTop:10,
		marginBottom:20,
		alignSelf:"center",
		alignContent:"center",
		alignItems:"center",
		justifyContent:"center",
		width:50,
		height:50,
		borderWidth:StyleSheet.hairlineWidth,
		borderColor:"#FFFFFF55",
	},
	modelButton:{
		backgroundColor:"#33333355",
		paddingBottom:10,
	},
	modelTitle:{
		color:'rgb(1,170,250)',
		fontSize:30,
		fontWeight:'bold',
		textAlign:'center'
	},
	imageElement:{
		opacity:1,
		borderWidth:StyleSheet.hairlineWidth,
		borderColor:"#FFFFFF55",
	},
	removeImage:{
		position:"absolute",
		width:20,
		height:20,
		fontSize:20,
		backgroundColor:"crimson",
		alignContent:"center",
		textAlign:"center",
		borderRadius:50,
		right:0
	},
	imagesCount:{
		position:'absolute',
		bottom:40,
		fontSize:15,
		width:50,
		height:40,
		fontWeight:"bold",
		borderRadius:100,
		alignSelf:"center",
		alignContent:"center",
		justifyContent:"center",
		textAlign:"center",
		textAlignVertical:"center",
		opacity:0.5,
	},
	button:{
		position:'absolute',
		bottom:30,
		width:50,
		height:50,
		alignItems:"center",
		justifyContent:"center",
		borderRadius:100,
		backgroundColor:"rgb(1,175,250)",
	},
	backButton:{
		left:20,
		alignSelf:"flex-start",
	},
	submitButton:{
		right:20,
		alignSelf:"flex-end",
	},
	submitButtonIcon:{
		fontSize:25,
		fontWeight:'900',
		color:"#111111",
	},
});
