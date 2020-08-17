import * as React from 'react';
import { Ionicons } from '@expo/vector-icons';
import { View, StyleSheet, PixelRatio,TouchableOpacity, Text, ActivityIndicator } from 'react-native';


import { ExpoWebGLRenderingContext, GLView } from 'expo-gl';
import ExpoTHREE, { Renderer, TextureLoader } from 'expo-three';
import OrbitControlsView from 'expo-three-orbit-controls';
import {
  AmbientLight,
  BoxBufferGeometry,
  Fog,
  GridHelper,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  PointLight,
  Scene,
  SpotLight,
} from 'three';

export default class ModelScreen extends React.Component {
	constructor(props){
		super(props)
		this.state = {
			loadComplete:false,
			camera:null,
			model:null,
		}
		this.onContextCreate = this.onContextCreate.bind(this)
	}

	componentDidMount(){
		THREE.suppressExpoWarnings(true)
		clearTimeout(this.timeout)
	}
	componentWillUnmount(){
		clearTimeout(this.timeout)
	}	

  	onContextCreate = async (gl: ExpoWebGLRenderingContext) => {
		const { drawingBufferWidth: width, drawingBufferHeight: height } = gl;
		const sceneColor = 'rgb(1,175,250)';
		
		const { uri, localUri, mWidth, mHeight } = await this.state.model.takeSnapshotAsync(gl,
			{'format':'png'})
		this.props.setSnap(uri)

		console.log({ uri, localUri, mWidth, mHeight })

		// Create a WebGLRenderer without a DOM element
		const renderer = new Renderer({ gl });
		renderer.setSize(width, height);
		renderer.setClearColor(sceneColor);

		const camera = new PerspectiveCamera(90, width / height, 0.01, 1000);
		camera.position.set(7, 3, 4);

		this.setState({camera:camera});

		const scene = new Scene();
		scene.fog = new Fog(sceneColor, 1, 10000);
		scene.add(new GridHelper(30, 30));

		const ambientLight = new AmbientLight(0x101010);
		scene.add(ambientLight);

		const pointLight = new PointLight(0xffffff, 2, 1000, 1);
		pointLight.position.set(0, 0, 500);
		scene.add(pointLight);

		const spotLight = new SpotLight(0xffffff, 0.5);
		spotLight.position.set(10, 10, 0);
		spotLight.lookAt(scene.position);
		scene.add(spotLight);
		
		// const cube = new IconMesh();
		// cube.position.set(2,2,2)
		// scene.add(cube);

		var model = {
			'obj': require('./../assets/models/toy.obj'),
			// 'png': require('./../assets/models/Hamburger/Hamburger.png'),
		};
		// Load model!
		var object = await ExpoTHREE.loadAsync(
			[model['obj']],
			null,
			name => model[name]
		);

		// Update size and position
		ExpoTHREE.utils.scaleLongestSideToSize(object, 1);
		ExpoTHREE.utils.alignMesh(object, { y: 1 });
		// Smooth mesh
		ExpoTHREE.utils.computeMeshNormals(object);
		object.scale.set(0.75,0.75,0.75);
		object.position.set(1,1,-1)
		object.rotation.set(-Math.PI/2,-Math.PI/2,-Math.PI)
		scene.add(object);

		// model = {
		// 	'obj': require('./../assets/models/thomas/thomas.obj'),
		// 	'mtl': require('./../assets/models/thomas/thomas.mtl'),
		// 	'png': require('./../assets/models/thomas/thomas.png'),
		// };
		// object = await ExpoTHREE.loadAsync(
		// 	[model['obj']],
		// 	null,
		// 	name => model[name]
		// );
		// // Update size and position
		// ExpoTHREE.utils.scaleLongestSideToSize(object, 1);
		// ExpoTHREE.utils.alignMesh(object, { y: 1 });
		// // Smooth mesh
		// ExpoTHREE.utils.computeMeshNormals(object);
		// object.scale.set(0.2,0.2,0.2);
		// object.position.x = 0
		// object.position.y = 0
		// object.position.z = 0
		// scene.add(object);
		
		camera.lookAt(scene.position);
		this.setState({loadComplete:true})

		function update() {
			// Cube
			// scene.children[4].rotation.x += 0.01;
			// scene.children[4].rotation.y += 0.01;
			// scene.children[4].rotation.z += 0.05;

			// // Hamburger
			// scene.children[5].rotation.y += 0.05;
			
			// // Train
			// scene.children[6].position.z += 0.01;
		}

		// Setup an animation loop
		const render = () => {
			this.timeout = requestAnimationFrame(render);
			update();
			renderer.render(scene, camera);

			// ref.current.getControls()?.update();
			gl.endFrameEXP();
		};
		render();
	};

	render(){
		return (
			<View style={styles.container}>
				<Text style={styles.previewTitle}>Preview</Text>
				<TouchableOpacity style={[styles.backButton]} activeOpacity={0.8}
					onPress= {()=>{
						this.props.setViewModel(false);
					}}
					>
					<Ionicons style={styles.backButtonIcon} name="md-arrow-back"></Ionicons>
				</TouchableOpacity>
				<OrbitControlsView style={styles.viewer} camera={this.state.camera}>
					<GLView style={{flex:1}} onContextCreate={this.onContextCreate} ref={ref => { this.state.model = ref; }} key="d" />
				</OrbitControlsView>
			</View>
		);
	}
}

class IconMesh extends Mesh {
  constructor() {
    super(
      new BoxBufferGeometry(0.5, 0.5, 0.5),
      new MeshStandardMaterial({
        map: new TextureLoader().load(require('./../assets/images/icon.png')),
      })
    );
  }
}

const styles = StyleSheet.create({
	container: {
		flex:1,
		alignItems:"center",
		backgroundColor:"#111111",
	},
	previewTitle:{
		marginVertical:10,
		height:50,
		fontSize:20,
        alignSelf: 'center',
        color:'rgb(1,175,250)',
        fontWeight: 'bold',
		backgroundColor: '#111111',
		alignContent:"center",
		justifyContent:"center",
		alignItems:"center",
		textAlign:"center",
		textAlignVertical:"center"
	},
	backButton:{
		position:'absolute',
		top:10,
		width:50,
		height:50,
		alignItems:"center",
		justifyContent:"center",
		borderRadius:100,
		backgroundColor:"rgb(1,175,250)",
		left:20,
		alignSelf:"flex-start",
		zIndex:10,
	},
	backButtonIcon:{
		fontSize:25,
		fontWeight:'900',
		color:"#111111",
	},
	viewer:{
		backgroundColor:"white",
		width:"100%",
		height:"100%",
	}
});