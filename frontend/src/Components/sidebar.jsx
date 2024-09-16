import React from 'react'
import '../Components/sidebar.css'
import { Outlet, useNavigate } from 'react-router-dom'

function Sidebars() {
    const navigate = useNavigate()
    const registerPage = () => {
        navigate('/sidebar/registerpage')
    }
    const identifyPage = () => {
        navigate('/sidebar/identifyPage')
    }
    const logOut = () => {
        navigate('/',{replace:true})
    }

    return (
        <>
            <div className='main-div'>
                <div className="sidebar">
                    <span className='span-button mt-3' onClick={logOut}>Logout</span>
                    <div className='sidebar-align'>
                        <span className='span-button ' onClick={registerPage}>vechicle Register</span>
                        <span className='span-button ' onClick={identifyPage}>Monitor vechicle</span>
                    </div>
                </div>

                <div className='right-side-div'>
                    <Outlet />

                </div>
            </div>
        </>

    )
}

export default Sidebars