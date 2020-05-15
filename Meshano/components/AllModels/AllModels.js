import React, { Component } from 'react'
import { ActivityIndicator, StyleSheet, View, AppRegistry,Image,Text,TouchableHighlight} from 'react-native'
import Constants from 'expo-constants';



export default class AllModels extends Component {
    constructor(){
        super()
        this.renderRow = this.renderRow.bind(this)
        this.state = {
            models:[
                {
                    title:"Model 1",
                    img:'./../../assets/icon.png'
                },
                // {
                //     title:"Model 2",
                //     img:"./../../assets/icon.png"
                // },
                // {
                //     title:"Model 3",
                //     img:"./../../assets/icon.png"
                // }
            ]
        }
    }

    renderRow = (models)=>{
        var row = [];
        models.forEach(model=>{
            row.push(
                <TouchableHighlight style={styles.modelButton}>
                    <View>
                        <Image style={styles.modelImage} source={require("./../../assets/icon.png")}></Image>
                        <Text style={{color:"white",marginTop:10,fontSize:16,textAlign:"center"}}>{model['title']}</Text>
                    </View>
                </TouchableHighlight>
            )
        });
        return row;
    }

    render() {
        return (
            <View style={styles.container}>
                <View style={styles.row}>
                    {this.renderRow(this.state.models)}
                </View>
            </View>
        )
    }
}

const styles = StyleSheet.create({
    container:{
        flex:1,
        flexDirection:"row",
        alignItems:"flex-start",
        backgroundColor:"#111111",
		marginTop:Constants.statusBarHeight,
    },
    row:{
        flexDirection:"row",
        flex:1,
        backgroundColor:"red"
    },
    modelButton:{
        // backgroundColor:"rgba(255,255,255,0.5)",
        width:150,
        height:150,
        backgroundColor:"white"
    },
    modelImage:{
        width:"100%",
        height:"100%",
        backgroundColor:"yellow"
    },
})
AppRegistry.registerComponent('AllModels',()=>AllModels);