import logo from './logo.svg';
import './App.css';
import Sidebars from './Components/sidebar';
import RegisterVechicle from './Components/RegisterVechicle';
import IdentifyVechicle from './Components/IdentifyVechicle';
import { BrowserRouter as Router, Routes, Route ,useHistory} from 'react-router-dom';
import Login from './Pages/Login';
import { useEffect } from 'react';


function App() {
  
  return (
    <div className="App">
      <Router>
        {/* <Sidebars/> */}
        <Routes>

          <Route path='/' exact element={<Login />} />
          <Route path='/sidebar' element={<Sidebars />}>
            {/* <Route index element={<RegisterVechicle/>}></Route> */}
            <Route path='/sidebar/registerpage' element={<RegisterVechicle />} />
            <Route path='/sidebar/identifypage' element={<IdentifyVechicle />} />
          </Route>
        </Routes>
      </Router>
    </div>
  );
}

export default App;

