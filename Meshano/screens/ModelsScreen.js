import { Ionicons } from '@expo/vector-icons';
import * as WebBrowser from 'expo-web-browser';
import * as React from 'react';
import { StyleSheet, Dimensions, Image, Text, View, TouchableOpacity, Modal, SafeAreaView } from 'react-native';
import { RectButton, ScrollView } from 'react-native-gesture-handler';
import ModelScreen from './ModelScreen'

const windowWidth = Dimensions.get('window').width;
var MODELS_PER_ROW = 2;

export default class ModelsScreen extends React.Component{

	constructor(props){
		super(props)
		this.state = {
			viewModel:false,
			isFocused:true,
			models:[
				{
					name:'Model 1',
					path:require('./../assets/images/icon.png')
				},{
					name:'Model 2',
					path:require('./../assets/images/icon.png')
			
				},{
					name:'Model 3',
					path:require('./../assets/images/icon.png')
			
				},{
					name:'Model 4',
					path:require('./../assets/images/icon.png')
			
				},{
					name:'Model 5',
					path:require('./../assets/images/icon.png')
			
				},{
					name:'Model 6',
					path:require('./../assets/images/icon.png')
			
				},{
					name:'Model 7',
					path:require('./../assets/images/icon.png')
			
				},
			],
			selectedModel:null,
			uri:null
		}

		this.calculatedSize = this.calculatedSize.bind(this) 
		this.scale = this.scale.bind(this) 
		this.renderRow = this.renderRow.bind(this) 
		this.renderRow = this.renderRow.bind(this) 
		this.setViewModel = this.setViewModel.bind(this)
		this.setSnap = this.setSnap.bind(this)
		
		this.props.navigation.addListener('focus', () => {
			this.setState({isFocused:true});
		});
		this.props.navigation.addListener('blur', () => {
			this.setState({isFocused:false});
		});
	}

	setViewModel = (viewModel)=>{
		this.setState({viewModel:viewModel})
	}

	setSnap = (uri)=>{
		this.setState({uri:uri})
		console.log("BARRA",this.state.uri)
	}

	calculatedSize(){
		var size = windowWidth / MODELS_PER_ROW
		return {width: size, height: size}
	}
	
	scale(){
		return {transform:[{scale:0.9}]}
	}
	
	renderRow(models){
		return models.map((model,i)=>{
			return (
				<TouchableOpacity key={i} style={[styles.modelButton, this.scale()]} activeOpacity={0.6}
					onPress={()=>{
						this.setViewModel(true)
						this.state.selectedModel = model
					}}
				>
					<Image source={{uri:this.state.uri}} style={[styles.modelImage, this.calculatedSize(), this.scale()]}></Image>
					{/* <Text style={styles.modelTitle}>{model.name}</Text> */}
				</TouchableOpacity>
			)
		});
	}
	renderRows(models) {
	
		let numberOfRows = Math.floor(models.length / MODELS_PER_ROW);
		let rows = []
		for (let i = 0; i < numberOfRows; i++) {
			let modelsForRow = models.slice(i*MODELS_PER_ROW,MODELS_PER_ROW*(i+1))
			rows.push(
				<View style={styles.row} key={i}>
					{this.renderRow(modelsForRow)}
				</View>
			)
		}
	
		if(models.length%MODELS_PER_ROW != 0){
			let modelsForRow = models.slice(numberOfRows*MODELS_PER_ROW,models.length)
			rows.push(
				<View style={styles.row} key={numberOfRows*MODELS_PER_ROW}>
					{this.renderRow(modelsForRow)}
				</View>
			)
		}
		return rows;
	}

	render(){
		return (
			this.state.isFocused && (
				<View style={styles.container}>
					<ScrollView style={{position:"relative",flex:1}}>
						{this.renderRows(this.state.models)}
					</ScrollView>
					<Modal
						visible={this.state.viewModel}
						animationType="slide"
						transparent={true}
						onRequestClose = {()=>{
							this.setState({viewModel:false})
						}}
					>
						<ModelScreen model={this.state.selectedModel} setSnap={this.setSnap} setViewModel={this.setViewModel}></ModelScreen>
					</Modal>
				</View>
			)
		);
	}
}

const styles = StyleSheet.create({
  container: {
	flex: 1,
	flexDirection:'column',
    backgroundColor: '#111111',
  },
  row:{
	flexDirection:'row',
  },
  modelButton:{
	backgroundColor:"#33333355",
	paddingBottom:10,
  },
  modelTitle:{
	color:'rgb(1,170,250)',
	fontSize:20,
	fontWeight:'bold',
	textAlign:'center'
  },
  modelImage:{
	opacity:1,
  },
  contentContainer: {
    paddingTop: 15,
  },
  optionIconContainer: {
    marginRight: 12,
  },
  option: {
    backgroundColor: '#fdfdfd',
    paddingHorizontal: 15,
    paddingVertical: 15,
    borderWidth: StyleSheet.hairlineWidth,
    borderBottomWidth: 0,
    borderColor: '#ededed',
  },
  lastOption: {
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  optionText: {
    fontSize: 15,
    alignSelf: 'flex-start',
    marginTop: 1,
  },
});
