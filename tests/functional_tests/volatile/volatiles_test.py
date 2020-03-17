from __future__ import absolute_import
""" Class description goes here. """

""" 
Use absolute imports. By default, when you import a package in Python 2, it adds '.' in the beggining, it means all 
imports are relative. It's a problem for our testing since our test can be in a package but our stubs are not including 
the package of the test (and shouldn't!). With __future__ we add Python 3 functionality for all imports to be absolute. 
"""
# Initialize dataClay
import os
import traceback
from mock.mockdataclay import MockDataClay
import unittest 
import logging
import pytest

"""
Volatiles testing. 
"""
logger = logging.getLogger(__name__)


class VolatilesTestCase(unittest.TestCase):

    """
    DataClayMock object for simulation. 
    """
    mock = None 
    """
    Indicates session is initialized.
    """
    session_initialized = False
    """ 
    Number of nodes in this test
    """ 
    num_nodes = 2

    def setUp(self):
        """
        PyUnit function called before every test case.
        Starts DataClay simulation in one Python interpreter and one Java VM. This allows us to Debug in a local machine without 
        dockers and without a full start of DataClay (jars, configurations, ...) 
        """ 
        try:
            self.session_initialized = False
            self.mock = MockDataClay(self.num_nodes) 
            self.mock.startSimulation(__file__)            
            self.mock.newAccount("bsc", "password")
            self.mock.newDataContract("bsc", "password", "dstest", "bsc")
            self.mock.newNamespace("bsc", "password", "model", "python")
            
            class_path = os.path.dirname(os.path.abspath(__file__)) + "/model"
            stubs_path = os.path.dirname(os.path.abspath(__file__)) + "/stubs"
    
            contractid = self.mock.newModel("bsc", "password", "model", class_path)
            self.mock.getStubs("bsc", "password", contractid, stubs_path)
            
            dataclay_client_config = os.environ["DATACLAYCLIENTCONFIG"]
            self.mock.prepareSessionFiles("bsc", "password", stubs_path, "dstest", "dstest", dataclay_client_config, "DS1")
        except:
            traceback.print_exc()

    def tearDown(self):
        """ 
        Finish all services started for simulation. 
        """ 
        if self.session_initialized:
            from dataclay.api import finish
            finish()
        self.mock.finishSimulation()
        logger.debug("Finished tear down of test")
    
    @pytest.mark.timeout(300, method='thread')
    def test(self):
        """Test. note that all test method names must begin with 'test.'"""
        """WARNING: IT IS HIGHLY RECOMMENDED TO HAVE ONE TEST ONLY TO ISOLATE FUNCTIONAL TESTS FROM EACH OTHER. i.e. 
        Start a new Python Interpreter and JVM for each test. In the end, it means only one test in this class. """
        
        from dataclay.api import init
    
        logger.debug('**Starting init**')
        init()
        
        """ 
        Imports. Imports must be located here in order to simulate "import" order in a real scenario. 
        VERY IMPORTANT: Imports must be located AFTER init
        """
        from model.classes import WebSite, WebPage, URI
        
        self.session_initialized = True
    
        """
        Test. From now on, the Functional Test itself. 
        """

        # Test recursive makePersistent without circular dependencies
        host = "bsc1.es"
        web_site = WebSite(host)
        web_site.make_persistent(alias=web_site.uri.host)
            
        uri = URI(host + "/volatilepage.html")  # Volatile
        web_site.uri = uri  # param of setter is volatile
                
        self.assertTrue(uri.is_persistent())
        self.assertTrue(web_site.is_persistent())
        self.assertTrue(web_site.uri.is_persistent())
                
        # Test recursive with one circular dependency
                
        host = "bsc2.es"
        web_page = WebPage(host + "/foo/bsc.html")
        web_page.make_persistent(alias=web_page.uri.host)
            
        host = "fsf.org"
        web_site = WebSite(host)
        web_site.add_web_page(web_page)  # added persistent object to a volatile 
                
        web_page.add_link(web_site)  # send volatile
                
        self.assertTrue(web_site.is_persistent())
        self.assertTrue(web_site.uri.is_persistent())
        self.assertTrue(web_page.is_persistent())
        self.assertTrue(web_page.uri.is_persistent())
        
        logger.debug("Test OK!")
