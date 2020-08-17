import * as React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { StyleSheet } from 'react-native';
import CameraScreen from '../screens/CameraScreen';
import PreviewScreen from '../screens/PreviewScreen';

const Stack = createStackNavigator();
const INITIAL_ROUTE_NAME = 'Camera';

export default class CreateScreen extends React.Component {
	constructor(props) {
		super(props);
		this.state={
			loaded:true,
			capturedImages : Array(0),
		}
		this.getImages = this.getImages.bind(this)
		this.clearImages = this.clearImages.bind(this)

		this.props.navigation.addListener('focus', () => {
			this.setState({loaded:true});
		});
		this.props.navigation.addListener('blur', () => {
			this.setState({loaded:false});
		});
	}

	componentDidMount(){
		this.setState({capturedImages:Array(0)})
	}

	componentWillUnmount(){
		this.setState({capturedImages:Array(0)})
	}

	getImages = (image)=>{
		this.setState({capturedImages:this.state.capturedImages.concat(image)})
	}
	clearImages = ()=>{
		this.setState({capturedImages:Array(0)})
		console.log(this.state.capturedImages.length)
	}

	render() {
			return (
				<Stack.Navigator
					style={styles.container}
					initialRouteName={INITIAL_ROUTE_NAME}
					screenOptions={{
					headerTitleAlign: 'center',
					headerStyle: {
						backgroundColor: '#621FF7',
					},
					headerTintColor: '#fff',
					headerTitleStyle :{
						fontWeight: 'bold',
					},
					}}
				>
					<Stack.Screen 
						name="Camera"
						component={CameraScreen} 
						options={{ 
							title: 'Camera'
						}}
						initialParams= {{
							capturedImages: this.state.capturedImages,
							addNewImage: this.getImages,
						}}
					/>
					<Stack.Screen
						name="Preview"
						component={PreviewScreen}
						options={{ 
							title: 'Preview'
						}}
						initialParams= {{
							capturedImages: this.state.capturedImages,
							clearImages: this.clearImages
						}}
					/>
				</Stack.Navigator>
			);
	}
}

const styles = StyleSheet.create({
	container: {
		flex:1,
		justifyContent:"center",
		alignItems:"center",
		backgroundColor:"#111111",
	},
	viewer:{
		width:"80%",
		height:"80%",
	}
});