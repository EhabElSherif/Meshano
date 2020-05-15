import React, { Component } from 'react'
import { Text, StyleSheet, View,Dimensions,TouchableOpacity,AppRegistry } from 'react-native';
import {Ionicons,FontAwesome} from '@expo/vector-icons';

export default class TabBar extends Component {
	constructor(props){
		super(props)
		this.route = this.route.bind(this)
	}

	route = (screenName)=>{
		console.log(this.props.stack)
		// this.props.stack(screenName)
	}

	render() {
        return (
            <View style={styles.tabBar}>
                <TouchableOpacity activeOpacity={0.5} style={styles.tabButton} onPress={()=>this.route('Camera')}>
					<Ionicons name="md-add" size={32} color="#111111" />
					{/* <Text style={styles.tabButtonText}>Create</Text> */}
				</TouchableOpacity>
                <TouchableOpacity activeOpacity={0.5} style={styles.tabButton} onPress={()=>this.route('Home')}>
					<Ionicons name="md-home" size={32} color="#111111" />
					{/* <Text style={styles.tabButtonText}>Home</Text> */}
				</TouchableOpacity>
                <TouchableOpacity activeOpacity={0.5} style={styles.tabButton} onPress={()=>this.route("AllModels")}>
					<Ionicons name="md-photos" size={32} color="#111111" />
					{/* <Text style={styles.tabButtonText}>View</Text> */}
				</TouchableOpacity>
            </View>
        )
    }
}

const styles = StyleSheet.create({
	tabBar:{
		position:"absolute",
		top:Dimensions.get("window").height-60,
		flexDirection:"row",
		width:"100%",
		height:60,
		backgroundColor:"rgb(0,125,190)"
	},
	tabButton:{
		flexGrow:1,
		alignItems:'center',
		backgroundColor:"rgb(1,175,250)",
		height:"100%",
		justifyContent:"center",
	},
	tabButtonText:{
		fontSize:25,
		color:"#111111"
	}
})
AppRegistry.registerComponent('TabBar',()=>TabBar);