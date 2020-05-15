import React, { Component } from 'react'
import { StyleSheet, Image, AppRegistry,Text, SafeAreaView } from 'react-native'

export default class Home extends Component {
	constructor(){
		super();
	}

    render() {
        return (
            <SafeAreaView style={{
				flex:1,
				justifyContent:'center',
				alignItems:'center',
				backgroundColor: "#111111",
				}}>
                <Image style={styles.logo} source={require('./../../assets/icon.png')}></Image>
				<Text style={{color:"rgb(1,175,250)",fontWeight:"bold",fontSize:40}}>من برة هاللا هاللا </Text>
				<Text style={{color:"rgb(1,175,250)",fontWeight:"bold",fontSize:20}}>و من جوا يعلم الله</Text>
            </SafeAreaView>
        )
    }
}


const styles = StyleSheet.create({
  	logo:{
		width:200,
		height:200,
	}
});
AppRegistry.registerComponent('Home',()=>Home);